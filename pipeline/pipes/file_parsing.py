import os
from . import Target
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox


class FileParser(Target):
    def __init__(self, data_directory):
        super().__init__()
        self.data_directory = data_directory

    def __call__(self, document):
        assert type(document) == str, f"input to file parser has wrong type: {type(document)}"
        assert document.endswith(".pdf") or document.endswith(".txt"), "invalid file path / type"

        document_name = self.get_document_name(document)
        return {
            "name": document_name,
            "pdf_structure": self.get_pdf_structure(document_name),
            "text": self.get_text(document_name)
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
                return {}

            structure = {}
            for index, page in enumerate(PDFPage.create_pages(doc)):
                try:
                    interpreter.process_page(page)
                    layout = device.get_result()
                    structure[f"{index}_{type(page).__name__}"] = self.parse_layout(layout)
                except:
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
