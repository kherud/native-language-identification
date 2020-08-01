import os
import re
import logging
import pickle
import pandas as pd
from . import Target
from collections import defaultdict


class FileParser(Target):
    def __init__(self,
                 data_directory,
                 conferences_file_path: str = os.path.abspath("pipeline/conferences.pkl")):
        super().__init__()
        self.data_directory = data_directory

        try:
            assert os.path.exists(conferences_file_path), f"conference data '{conferences_file_path}' does not exist"
            with open(conferences_file_path, "rb") as file:
                self.conferences = pickle.load(file)
        except AssertionError:
            logging.error(f"{conferences_file_path} not found")
            self.conferences = {}

        self.abstract_re = re.compile("\s*".join("ABSTRACT") + "|" + "\s*".join("Abstract"))

    def __call__(self, document):
        assert type(document) == str, f"input to file parser has wrong type: {type(document)}"
        assert document.endswith(".txt"), "invalid file path / type, expected .txt"

        document_name = self.get_document_name(document)
        document_text = self.get_text(document)

        if document_name in self.conferences:
            meta = self.conferences[document_name]
        else:
            meta = {}
            logging.error(f"{document_name} not found in conferences file")

        abstract_start, abstract_end = self.detect_abstract(document_text, meta)

        return {
            "name": document_name,
            "meta": meta,
            "abstract_start": abstract_start,
            "abstract_end": abstract_end,
            # "pdf_structure": self.get_pdf_structure(document_name),
            "text": document_text,
            "text_cleaned": document_text,
            "entities": defaultdict(set),
        }

    def get_document_name(self, document):
        return document.split("/")[-1].split(".")[0]

    def detect_abstract(self, text, meta):
        abstract_mention = None
        abstract_title_mentions = []
        if "title" in meta:
            abstract_title_mentions = self.abstract_re.findall(meta["title"])
        if len(abstract_title_mentions) > 0:
            abstract_mentions = list(self.abstract_re.finditer(text))
            if len(abstract_mentions) >= len(abstract_title_mentions):
                abstract_mention = abstract_mentions[len(abstract_title_mentions)]
        else:
            abstract_mention = self.abstract_re.search(text)

        if abstract_mention is None:
            return 800, 800  # mean based default value
        else:
            start = min(abstract_mention.start(), 2500)
            end = min(abstract_mention.end(), 2500)
            return start, end  # min due to error tolerance

    def get_text(self, document_path):
        with open(document_path, "r") as file:
            return file.read()


class CsvWriter(Target):
    def __init__(self, data_directory, output_dir="csvs"):
        super().__init__()
        self.output_path = os.path.join(os.path.dirname(data_directory), output_dir)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def __call__(self, document):
        assert type(document) == dict, f"input to file parser has wrong type: {type(document)}"

        result = {
            "Text": [],
            "Label": [],
        }

        for entity_name, entities in document["entities"].items():
            for entity in entities:
                result["Text"].append(entity)
                result["Label"].append(str(entity_name))

        file_path = os.path.join(self.output_path, document["name"] + ".csv")

        pd.DataFrame(result).to_csv(file_path, index=False)

        document["result"] = result

        return document


class TextWriter(Target):
    def __init__(self, data_directory, output_dir="txts_cleaned"):
        super().__init__()
        self.output_path = os.path.join(os.path.dirname(data_directory), output_dir)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def __call__(self, document):
        assert type(document) == dict, f"input to file parser has wrong type: {type(document)}"

        file_path = os.path.join(self.output_path, document["name"] + ".txt")

        with open(file_path, "w") as file:
            file.write(document["text_cleaned"])

        return document
