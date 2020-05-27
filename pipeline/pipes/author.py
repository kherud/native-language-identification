import os
import spacy
from . import Target, Entity


class AuthorParser(Target):
    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = os.path.abspath(model_dir)
        assert os.path.exists(self.model_dir), f"ner model directory '{self.model_dir}' does not exist"

        self.ner = spacy.load(self.model_dir)

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to author parser"

        result = self.ner(document["text"][:document["abstract_start"]])

        for ent in result.ents:
            if ent.label_ == "PERSON":
                document["entities"][Entity.AUTHOR].add(ent.text)
            if ent.label_ == "ORG":
                document["entities"][Entity.INSTITUTION_COMPANY].add(ent.text)

        return document
