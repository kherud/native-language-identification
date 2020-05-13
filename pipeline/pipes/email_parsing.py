import re
from collections import defaultdict
from . import Target


class EmailExtractor(Target):
    def __init__(self):
        super().__init__()
        self.re_email = re.compile(r'[\w\.-]+@[\w\.-]+')

    def process(self, value):
        assert isinstance(value, dict), f"wrong input of type {type(value)} to email extractor"
        assert "text" in value, "no text in document"

        if "entities" not in value:
            value["entities"] = defaultdict(set)

        for email in self.re_email.findall(value["text"]):
            value["entities"]["email"].add(email)

        return value
