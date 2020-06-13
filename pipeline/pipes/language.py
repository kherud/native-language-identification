import re
import os
import spacy
import logging
import unicodedata

from . import Target, Entity
from spacy.lang.en import English
from spacy_cld import LanguageDetector

"""
https://spacy.io/universe/project/spacy_cld
"""


class LanguageParser(Target):
    def __init__(self, sentencizer_dir: str = os.path.abspath("models/sentencizer")):
        super().__init__()
        self.language_detector = English()
        self.language_detector.add_pipe(LanguageDetector())
        self.relevant_languages = {"ar", "zh", "fr", "de", "es", "it", "ja", "ru",
                                   "tk", "vi", "pl", "ko", "cs", "or", "syr", "zh-Hant"}

        self.sentencizer_dir = os.path.abspath(sentencizer_dir)
        assert os.path.exists(self.sentencizer_dir), f"ner model directory '{self.sentencizer_dir}' does not exist"
        self.sentencizer = spacy.load(self.sentencizer_dir)

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to location parser"

        noise_re = r"et\.?\sal\.?|et\.?\s|al\."
        if "references_authors" in document:
            noise_re += "|".join(re.escape(x) for x in document["references_authors"])
        noise_re = re.compile(noise_re, re.IGNORECASE)

        sentences = self.get_sentences(document)
        for sentence in sentences:
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

            text = document["text_cleaned"][sentence.start_char:sentence.end_char]
            document["entities"][Entity.LANGUAGE].add(text)

            self.clean_text(document, sentence_text)

        return document

    def get_sentences(self, document):
        try:
            result = self.sentencizer(document["text_cleaned"])
            return result.sents
        except KeyError:
            logging.error(f"cannot parse sentences of '{document['name']}'")
            return []
