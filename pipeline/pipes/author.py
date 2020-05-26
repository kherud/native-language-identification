import re
import os
import spacy
from . import Target


class AuthorParser(Target):
    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = os.path.abspath(model_dir)
        self.ner = spacy.load(self.model_dir)
        self.abstract_re = re.compile("\s*".join("Abstract"))

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to author parser"

        abstract = self.abstract_re.search(document["text"])

        if not abstract or abstract.start() > 2000:
            abstract_start = 800
        else:
            abstract_start = abstract.start()

        text = document["text"][:abstract_start]

        result = self.ner(text)

        for ent in result.ents:
            if ent.label_ == "PERSON":
                document["entities"]["author"].add(ent.text)
            if ent.label_ == "ORG":
                document["entities"]["institution/company"].add(ent.text)

        return document
