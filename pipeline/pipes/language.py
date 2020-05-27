import re
import unicodedata

from . import Target, Entity
from spacy.lang.en import English
from spacy_cld import LanguageDetector

"""
https://spacy.io/universe/project/spacy_cld
"""


class LanguageParser(Target):
    def __init__(self):
        super().__init__()
        self.language_detector = English()
        self.language_detector.add_pipe(LanguageDetector())
        self.relevant_languages = {"ar", "zh", "fr", "de", "es", "it", "ja", "ru",
                                   "tk", "vi", "pl", "ko", "cs", "or", "syr", "zh-Hant"}

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to location parser"

        noise_re = r"et\.?\sal\.?|et\.?\s|al\."
        if "references_authors" in document:
            noise_re += "|".join(re.escape(x) for x in document["references_authors"])
        noise_re = re.compile(noise_re, re.IGNORECASE)

        for sentence in document["sentences"]:
            # some unicode characters break the library...
            sentence_text = "".join([c for c in sentence.text if unicodedata.category(c)[0] not in ('S', 'M', 'C')])
            sentence = self.language_detector(sentence_text)

            # continue if no language detected
            if len(sentence._.language_scores) == 0:
                continue
            language_max = max(sentence._.language_scores, key=sentence._.language_scores.get)
            # continue if language should not be removed
            if language_max not in self.relevant_languages:
                continue
            # continue if sentence is most likely reference
            if noise_re.search(sentence.text):
                continue

            document["entities"][Entity.LANGUAGE].add(sentence.text)

        return document
