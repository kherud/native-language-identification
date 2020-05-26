import re
import string
from . import Target


class EmailParser(Target):
    def __init__(self):
        super().__init__()
        chars = "A-Za-z\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF\-\."
        self.space_re = re.compile(r"[\s]+")
        self.noise_re = re.compile(f"[^{string.printable}\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF]")
        self.email_re = re.compile(rf"[{chars}]+@[{chars}]+(?:\.[{chars}]+)+")
        self.email_re2 = re.compile(rf"{{[{chars}\s]+}}@[{chars}]+(?:\.[{chars}]+)+")

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to email extractor"

        text = self.space_re.sub(" ", document["text"])  # replace multiple whitespaces with single space
        text = self.noise_re.sub("", text)

        for email in self.email_re.findall(text):
            document["entities"]["email"].add(email)
        for email in self.email_re2.findall(text):
            document["entities"]["email"].add(email)

        return document
