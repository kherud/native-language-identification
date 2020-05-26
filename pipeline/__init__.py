import os
import glob
import tqdm
from typing import List, Union
from pipeline.pipes import worker, Target
from pipeline.pipes.file import FileParser, CsvWriter
from pipeline.pipes.email import EmailParser
from pipeline.pipes.entity import EntityParser
from pipeline.pipes.author import AuthorParser
from pipeline.pipes.reference import ReferenceParser
from pipeline.pipes.geolocation import LocationParser
from pipeline.pipes.acknowledgement import AcknowledgementParser
from multiprocessing import Process, Pipe, Lock, Manager, Queue, cpu_count
from multiprocessing.connection import Connection


def parallel(constructor, kwargs, in_queue, out_queue, lock):
    target = constructor(**kwargs)
    while in_queue.qsize() > 0:
        with lock:
            document = in_queue.get()
        target(document)
        out_queue.put(1)


def process_with_pool(data_directory: str, **kwargs):
    """ Used to spawn multiple single process pipelines in different processes """
    in_queue = Queue()

    assert os.path.exists(data_directory), f"'{data_directory}' does not exists"

    pdfs_location = os.path.join(data_directory, "pdfs")
    assert os.path.exists(pdfs_location), f"'{pdfs_location}' does not exists"

    for document in glob.glob(f"{pdfs_location}/*.pdf"):
        in_queue.put(document)
    assert in_queue.qsize() > 0, f"no documents found in '{data_directory}/pdfs/'"

    lock = Lock()
    out_queue = Queue()

    constructor = PipelineSingleprocess if "pipeline" in kwargs else PipelineSingleprocess.factory
    kwargs["data_directory"] = data_directory

    n_processes = cpu_count()
    if "processes" in kwargs and kwargs["processes"] is not None:
        n_processes = kwargs["processes"]
    processes = [Process(target=parallel,
                          args=(constructor, kwargs, in_queue, out_queue, lock))
                  for _ in range(n_processes)]

    n_documents = in_queue.qsize()
    with tqdm.tqdm(total=n_documents, desc="Status: Loading Pipes...", unit="Documents") as progress_bar:
        for process in processes:
            process.start()

        while out_queue.get():
            progress_bar.set_description("Progress")
            progress_bar.update(1)
            if progress_bar.n == n_documents:
                break


class PipelineSingleprocess:
    """ Whole pipeline in one process """

    def __init__(self,
                 pipeline: List[Target],
                 data_directory: str):
        self.pipeline = pipeline
        self.data_directory = data_directory

    @staticmethod
    def factory(data_directory: str, ner_model: str = "models/ner", **kwargs):
        return PipelineSingleprocess(
            pipeline=[
                FileParser(data_directory),
                AuthorParser(ner_model),
                EmailParser(),
                LocationParser(),
                ReferenceParser(),
                AcknowledgementParser(),
                CsvWriter(data_directory),
            ],
            data_directory=data_directory
        )

    def __call__(self, document):
        assert len(self.pipeline) > 0, "no pipes in pipeline"
        for pipe in self.pipeline:
            document = pipe(document)
        return document


class PipelineMultiprocess:
    """ Pipeline with pipes distributed across different processes """

    def __init__(self,
                 pipeline: List[Pipe],
                 pipe_in: Union[Connection, Queue],
                 pipe_out: Connection,
                 data_directory: str):
        self.pipeline = pipeline
        self.pipe_in = pipe_in
        self.pipe_out = pipe_out
        self.data_directory = data_directory

    @staticmethod
    def factory(data_directory):
        input_queue = Queue()

        (p1_out, p1_in), p1_lock = Pipe(duplex=False), Lock()
        (p2_out, p2_in), p2_lock = Pipe(duplex=False), Lock()
        (p3_out, p3_in), p3_lock = Pipe(duplex=False), Lock()
        (p4_out, p4_in), p4_lock = Pipe(duplex=False), Lock()
        (p5_out, p5_in), p5_lock = Pipe(duplex=False), Lock()
        (p6_out, p6_in), p6_lock = Pipe(duplex=False), Lock()

        data_directory = os.path.abspath(data_directory)

        pipeline = [
            Process(target=worker, args=(FileParser(data_directory), input_queue, p2_in, p1_lock)),
            Process(target=worker, args=(FileParser(data_directory), input_queue, p2_in, p1_lock)),
            Process(target=worker, args=(FileParser(data_directory), input_queue, p2_in, p1_lock)),
            Process(target=worker, args=(EmailParser(), p2_out, p3_in, p2_lock)),
            Process(target=worker, args=(EntityParser(), p3_out, p4_in, p3_lock)),
            Process(target=worker, args=(ReferenceParser(), p4_out, p5_in, p4_lock)),
            Process(target=worker, args=(CsvWriter(data_directory), p5_out, p6_in, p5_lock)),
        ]

        return PipelineMultiprocess(pipeline, input_queue, p6_out, data_directory)

    def start(self, files_dir="pdfs"):
        file_paths = glob.glob(f"{self.data_directory}/{files_dir}/*.pdf")

        for file_path in file_paths:
            self.pipe_in.put(file_path)

        for pipe in self.pipeline:
            pipe.start()

        pbar = tqdm.tqdm(desc="Progress", unit="Documents", total=len(file_paths))
        while pbar.n < len(file_paths):
            try:
                self.pipe_out.recv()
            except Exception as e:
                print(e)
            pbar.update(1)

        for pipe in self.pipeline:
            pipe.terminate()
