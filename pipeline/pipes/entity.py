import spacy
from . import Target


RELEVANT_ENTITIES = ["PERSON", "ORG", "GPE", "NORP", "FAC"]

class EntityParser(Target):
    def __init__(self):
        super().__init__()
        raise DeprecationWarning()
        # self.spark = sparknlp.start()
        # self.ner_pipeline = PretrainedPipeline('recognize_entities_dl', 'en')
        self.ner_pipeline = spacy.load("en_core_web_md", disable=["tagger", "parser"])

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to entity parser"

        result = self.ner_pipeline(document["text"].replace("\n", " "))

        for entity in result.ents:
            if entity.label_ in RELEVANT_ENTITIES:
                document["entities"][entity.label_].add(entity.text)

        return document
