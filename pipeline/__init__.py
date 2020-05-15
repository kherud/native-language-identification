import os
import abc
import glob
import tqdm
from typing import List, Union
from pipeline.pipes import worker
from pipeline.pipes.file_parsing import FileParser
from pipeline.pipes.file_writing import CsvWriter
from pipeline.pipes.email_parsing import EmailParser
from pipeline.pipes.entity_parsing import EntityParser
from pipeline.pipes.reference_parsing import ReferenceParser
from multiprocessing import Process, Pipe, Lock, Manager, Queue
from multiprocessing.connection import Connection


class Pipeline:
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

        return Pipeline(pipeline, input_queue, p6_out, data_directory)

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
