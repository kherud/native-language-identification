import re
import string
from collections import defaultdict
from . import Target


class EmailParser(Target):
    def __init__(self):
        super().__init__()
        self.space_re = re.compile(r"[\s]+")
        self.noise_re = re.compile(f"[^{string.printable}\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF]")
        self.email_re = re.compile(r"[\w.-]+@[\w-]+\.[\w-]+")

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to email extractor"
        assert "text" in document, "no text in document"

        if "entities" not in document:
            document["entities"] = defaultdict(set)

        text = self.space_re.sub(" ", document["text"])  # replace multiple whitespaces with single space
        text = self.noise_re.sub("", text)

        for email in self.email_re.findall(text):
            document["entities"]["email"].add(email)

        return document
