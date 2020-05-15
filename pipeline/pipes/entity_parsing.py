# import sparknlp
# from sparknlp.pretrained import PretrainedPipeline
from . import Target
import spacy
from collections import defaultdict


RELEVANT_ENTITIES = ["PERSON", "ORG", "GPE", "NORP", "FAC"]


class EntityParser(Target):
    def __init__(self):
        super().__init__()
        # self.spark = sparknlp.start()
        # self.ner_pipeline = PretrainedPipeline('recognize_entities_dl', 'en')
        self.ner_pipeline = spacy.load("en_core_web_md", disable=["tagger", "parser"])

    def process(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to entity parser"
        assert "text" in document, "no text in document"

        if "entities" not in document:
            document["entities"] = defaultdict(set)

        result = self.ner_pipeline(document["text"].replace("\n", " "))

        for entity in result.ents:
            if entity.label_ in RELEVANT_ENTITIES:
                document["entities"][entity.label_].add(entity.text)

        # result = self.ner_pipeline.annotate(value["text"])
        #
        # entity = []
        # last_ner = ""
        # for token, ner in zip(result["token"], result["ner"]):
        #     if (len(entity) > 1 and ner.startswith("B-")) or (len(entity) > 0 and ner == "O"):
        #         ner_tag = last_ner.split("-")[-1]
        #         value["entities"][ner_tag].add(" ".join(entity).lower())
        #         entity = []
        #     if not ner == "O":
        #         entity.append(token)
        #     last_ner = ner

        return document
