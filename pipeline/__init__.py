import os
import glob
import tqdm
from typing import List, Union, Callable
from pipeline.pipes import worker, Target
from pipeline.pipes.acknowledgement import AcknowledgementParser
from pipeline.pipes.classification import LSTMClassification
# from pipeline.pipes.author import AuthorParser
from pipeline.pipes.email import EmailParser
# from pipeline.pipes.entity import EntityParser
from pipeline.pipes.file import FileParser, CsvWriter, TextWriter, FileReader, PredictionWriter
from pipeline.pipes.footnotes import FootnoteParser
from pipeline.pipes.geolocation import LocationParser
from pipeline.pipes.language import LanguageParser
from pipeline.pipes.pre_abstract import PreAbstractParser
from pipeline.pipes.reference import ReferenceParser
from pipeline.pipes.review import ReviewerParser
from pipeline.pipes.tokenization import BPETokenization
from multiprocessing import Process, Lock, Queue, cpu_count


def parallel(pipeline: Union[Callable, "Pipeline"],
             kwargs: dict,
             in_queue: Queue,
             out_queue: Queue,
             lock: Lock):
    target = pipeline(**kwargs)
    while in_queue.qsize() > 0:
        with lock:
            document = in_queue.get()
        target(document)
        out_queue.put(1)


def process_with_pool(data_directory: str,
                      pipeline: Callable,
                      **kwargs):
    """ Used to spawn multiple pipelines to distribute the workload across processes """
    in_queue = Queue()

    assert os.path.exists(data_directory), f"'{data_directory}' does not exists"

    for document in glob.glob(f"{data_directory}/*.txt"):
        in_queue.put(document)
    assert in_queue.qsize() > 0, f"no documents found in '{data_directory}'"

    lock = Lock()
    out_queue = Queue()

    kwargs["data_directory"] = data_directory

    n_processes = cpu_count()
    if "processes" in kwargs and kwargs["processes"] is not None:
        n_processes = kwargs["processes"]
    processes = [Process(target=parallel,
                         args=(pipeline, kwargs, in_queue, out_queue, lock))
                 for _ in range(n_processes)]

    n_documents = in_queue.qsize()
    with tqdm.tqdm(total=n_documents, desc="Status: Loading Pipes", unit="Documents") as progress_bar:
        for process in processes:
            process.start()

        while out_queue.get():
            progress_bar.set_description("Progress")
            progress_bar.update(1)
            if progress_bar.n == n_documents:
                break


class Pipeline:
    """ Whole pipeline in one process """

    def __init__(self,
                 pipeline: List[Target],
                 data_directory: str):
        self.pipeline = pipeline
        self.data_directory = data_directory

    def __call__(self, document):
        assert len(self.pipeline) > 0, "no pipes in pipeline"
        for pipe in self.pipeline:
            document = pipe(document)
        return document

    @staticmethod
    def preprocessing_factory(data_directory: str,
                              pre_abstract_model_dir: str = "models/pre_abstract/model",
                              references_model_dir: str = "models/references/model",
                              device: str = "cpu",
                              **kwargs):
        return Pipeline(
            pipeline=[
                FileParser(data_directory),
                PreAbstractParser(pre_abstract_model_dir, device=device),
                # AuthorParser(ner_model),
                ReferenceParser(references_model_dir, device=device),
                ReviewerParser(),
                AcknowledgementParser(),
                EmailParser(),
                LocationParser(),
                FootnoteParser(),
                LanguageParser(),
                CsvWriter(data_directory),
                TextWriter(data_directory),
            ],
            data_directory=data_directory
        )

    @staticmethod
    def classification_factory(data_directory: str,
                               classifier_dir: str = "models/classification/model",
                               tokenizer_dir: str = "models/classification/tokenizer",
                               device: str = "cpu",
                               **kwargs):
        return Pipeline(
            pipeline=[
                FileReader(data_directory),
                BPETokenization(tokenizer_dir),
                LSTMClassification(classifier_dir, device=device),
                PredictionWriter(data_directory),
            ],
            data_directory=data_directory
        )
