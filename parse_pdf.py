import glob
import json
import tqdm
import spacy
from multiprocessing import Pool, current_process
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTTextBoxHorizontal


nlp = spacy.load("en_core_web_md", disable=["tagger", "parser"])


def parse_pdf(file_name):
    print(current_process(), file_name)
    with open(file_name, "rb") as file:
        parser = PDFParser(file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        structure = {}
        for index, page in enumerate(PDFPage.create_pages(doc)):
            interpreter.process_page(page)
            layout = device.get_result()
            structure[f"{index}_{type(page).__name__}"] = parse_layout(layout)

    file_name = "data/jsons/" + "".join(file_name.split("/")[-1]) + ".json"
    with open(file_name, "w") as structure_file:
        json.dump(structure, structure_file)


def parse_layout(layout):
    if not isinstance(layout, LTTextBox):
        return {f"{index}_{type(obj).__name__}": parse_layout(obj)
                for index, obj in enumerate(layout)
                if hasattr(obj, "__iter__")}
    return layout.get_text()


if __name__ == "__main__":
    dir_name = "files"

    parse_pdf("data/pdfs/AAAI12-4.pdf")

    # files = glob.glob(f"{dir_name}/*.pdf")
    # with Pool() as pool:
    #     for _ in tqdm.tqdm(pool.imap_unordered(parse_pdf, files), total=len(files)):
    #         pass