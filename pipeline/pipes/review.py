import re
from . import Target, Entity


class ReviewerParser(Target):
    def __init__(self):
        super().__init__()
        self.mention_re = re.compile(r"Reviewed\s*by\n([^A-Za-z]\n)?[^\n]+\n([^A-Za-z]\n)?[^\n]+")

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to location parser"

        for mention in self.mention_re.findall(document["text_cleaned"]):
            document["entities"][Entity.REVIEWER].add(mention)
            self.clean_text(document, mention)

        return document
