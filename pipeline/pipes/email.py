import re
from . import Target, Entity


class EmailParser(Target):
    def __init__(self):
        super().__init__()
        self.email_re = re.compile(r"\S+@\S+(?:\.\S+)+")
        self.email_re2 = re.compile(r"{{.+}}@\S+(?:\.\S+)+")

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to email extractor"

        for email in self.email_re.findall(document["text"]):
            document["entities"][Entity.EMAIL].add(email)
        for email in self.email_re2.findall(document["text"]):
            document["entities"][Entity.EMAIL].add(email)

        return document
