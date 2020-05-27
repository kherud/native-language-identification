import re
from . import Target, Entity
from geotext import GeoText
from spacy.lang.en.stop_words import STOP_WORDS


class LocationParser(Target):
    def __init__(self):
        super().__init__()
        self.stop_words = STOP_WORDS
        self.stop_words.add("university")
        self.stop_words.add("central")

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to location parser"

        geo = GeoText(document["text"])

        for mention in geo.countries + geo.cities + geo.nationalities:
            if mention.lower() in self.stop_words:
                continue
            for match in re.finditer("[\s-]*".join(mention), document["text"], re.IGNORECASE):
                document["entities"][Entity.COUNTRY].add(document["text"][match.start():match.end()])

        return document
