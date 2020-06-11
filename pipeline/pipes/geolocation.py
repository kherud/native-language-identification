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

        geo = GeoText(document["text_cleaned"])
        for mention in geo.countries:  # geo.cities + geo.nationalities:
            if mention.lower() in self.stop_words:
                continue
            for match in re.finditer("[\s-]*".join(mention), document["text_cleaned"], re.IGNORECASE):
                country = document["text"][match.start():match.end()]
                # non capitalized words result in poor precision
                if country.capitalize() != country:
                    continue
                document["entities"][Entity.COUNTRY].add(country)
                self.clean_text(document, country, cased=True)


        # low precision -> UK, CH etc.
        # geo = GeoText(document["text"][:document["abstract_start"]])
        # for mention in geo.country_mentions:
        #     if mention.lower() in self.stop_words:
        #         continue
        #     for match in re.finditer(mention, document["text"], re.IGNORECASE):
        #         document["entities"][Entity.COUNTRY].add(document["text"][match.start():match.end()])

        return document
