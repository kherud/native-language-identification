import os
import re
import spacy
import logging
import pandas as pd
from . import Target
from collections import defaultdict
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox


class FileParser(Target):
    def __init__(self, data_directory, sentencizer_dir: str = "models/sentencizer"):
        super().__init__()
        self.data_directory = data_directory

        self.sentencizer_dir = os.path.abspath(sentencizer_dir)
        assert os.path.exists(self.sentencizer_dir), f"ner model directory '{self.sentencizer_dir}' does not exist"
        self.sentencizer = spacy.load(self.sentencizer_dir)

        self.abstract_re = re.compile("\s*".join("ABSTRACT") + "\s+|" + "\s*".join("Abstract") + "\s+")

    def __call__(self, document):
        assert type(document) == str, f"input to file parser has wrong type: {type(document)}"
        assert document.endswith(".pdf") or document.endswith(".txt"), "invalid file path / type"

        document_name = self.get_document_name(document)
        document_text = self.get_text(document_name)

        abstract_start, abstract_end = self.detect_abstract(document_text)

        return {
            "name": document_name,
            # "pdf_structure": self.get_pdf_structure(document_name),
            "text": document_text,
            "text_cleaned": document_text,
            "entities": defaultdict(set),
            "abstract_start": abstract_start,
            "abstract_end": abstract_end,
            "sentences": self.get_sentences(document_text[abstract_end:]),
        }

    def get_document_name(self, document):
        file_name = document.split("/")[-1]
        return file_name.replace(".pdf", "").replace(".txt", "")

    def get_pdf_structure(self, document_name):
        with open(f"{os.path.join(self.data_directory, 'pdfs', document_name)}.pdf", "rb") as file:
            parser = PDFParser(file)
            try:
                doc = PDFDocument(parser)
                rsrcmgr = PDFResourceManager()
                laparams = LAParams()
                device = PDFPageAggregator(rsrcmgr, laparams=laparams)
                interpreter = PDFPageInterpreter(rsrcmgr, device)
            except:
                logging.error(f"cannot parse pdf structure of '{document_name}'")
                return {}

            structure = {}
            for index, page in enumerate(PDFPage.create_pages(doc)):
                try:
                    interpreter.process_page(page)
                    layout = device.get_result()
                    structure[f"{index}_{type(page).__name__}"] = self.parse_layout(layout)
                except:
                    logging.error(f"cannot parse page {index+1} of '{document_name}.pdf'")
                    structure[f"{index}_{type(page).__name__}"] = ""

        return structure

    def parse_layout(self, layout):
        if not isinstance(layout, LTTextBox):
            return {f"{index}_{type(obj).__name__}": self.parse_layout(obj)
                    for index, obj in enumerate(layout)
                    if hasattr(obj, "__iter__")}
        return layout.get_text()

    def get_text(self, document_name):
        with open(f"{os.path.join(self.data_directory, 'txts', document_name)}.pdf.txt", "r") as file:
            return file.read()

    def detect_abstract(self, text):
        abstract = self.abstract_re.search(text)

        if not abstract or abstract.start() > 2000:
            logging.error("no valid abstract found")
            return 800, 800  # return 99% CI
        else:
            return abstract.start(), abstract.end()

    def get_sentences(self, text):
        try:
            result = self.sentencizer(text)
            return result.sents
        except KeyError:
            logging.error("cannot parse sentences")
            return []


class CsvWriter(Target):
    def __init__(self, data_directory, output_dir="csvs"):
        super().__init__()
        self.output_path = os.path.join(data_directory, output_dir)

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
        self.output_path = os.path.join(data_directory, output_dir)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def __call__(self, document):
        assert type(document) == dict, f"input to file parser has wrong type: {type(document)}"

        file_path = os.path.join(self.output_path, document["name"] + ".txt")

        with open(file_path, "w") as file:
            file.write(document["text_cleaned"])

        return document
