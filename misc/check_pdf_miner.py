import glob
import json
import tqdm
from multiprocessing import Pool, current_process
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator, TextConverter
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTTextBoxHorizontal
from io import StringIO


def get_pdf_text(filename):
    print(filename)
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    with open(filename, 'rb') as fd:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(fd):
            interpreter.process_page(page)
        text = retstr.getvalue()
    device.close()
    retstr.close()

def parse_layout(layout):
    if not isinstance(layout, LTTextBox):
        return {f"{index}_{type(obj).__name__}": parse_layout(obj)
                for index, obj in enumerate(layout)
                if hasattr(obj, "__iter__")}
    return layout.get_text()


if __name__ == "__main__":

    file_paths = glob.glob(f"../data/pdfs/*.pdf")
    with Pool() as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(get_pdf_text, file_paths), total=len(file_paths)):
            pass