import re
import os
import spacy
import logging
from . import Target, Entity


class FootnoteParser(Target):
    def __init__(self, sentencizer_dir: str = os.path.abspath("models/sentencizer")):
        super().__init__()
        self.key_words = ["sponsored", "funded", "funding", "internship"]
        noise_re = "\s*".join("Acknowledg") + "e?" + "\s*".join("ment") + "(?:\s*s)?\s*"
        noise_re += "|" + "\s*".join("ACKNOWLEDG") + "E?" + "\s*".join("MENT") + "(?:\s*S)?\s*"
        noise_re += "|" + "\s*".join("Reference") + "\s*s?\s*"
        noise_re += "|" + "\s*".join("REFERENCE") + "\s*S?\s*"
        noise_re += "|" + "\[\d+\]"
        self.noise_re = re.compile(noise_re)
        self.noise_re2 = re.compile(r"[^A-Za-z\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF,; ]", re.IGNORECASE)
        self.and_re = re.compile(r"and|&|;", re.IGNORECASE)
        self.keyword_re = re.compile(r"acknowledges?\s(?:the\s)?support|was\s(?:\w{1,10}\s)?supported\sby")

        self.sentencizer_dir = os.path.abspath(sentencizer_dir)
        assert os.path.exists(self.sentencizer_dir), f"ner model directory '{self.sentencizer_dir}' does not exist"
        self.sentencizer = spacy.load(self.sentencizer_dir)

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to location parser"

        key_words = set(self.key_words)
        try:
            for author in document["meta"]["authors"]:
                key_words.add(author)
            for org in document["meta"]["orgs"]:
                # only consider long keywords, since most short ones result in poor precision, e.g. google
                if len(org["name"]) > 10:
                    key_words.add(org["name"])
        except KeyError:
            logging.error(f"meta data key missing in '{document['name']}'")

        for authors in document["entities"][Entity.AUTHOR]:
            authors = self.and_re.sub(",", authors)
            authors = self.noise_re2.sub("", authors)
            for author in authors.split(","):
                if len(author) < 3:
                    continue
                key_words.add(author.strip())

        sentences = self.get_sentences(document)
        for sentence in sentences:
            if any(re.search(r"[\s\-]*".join(re.escape(k) for k in key_word.split()), sentence.text, re.IGNORECASE) for key_word in key_words) \
                    or self.keyword_re.search(sentence.text):
                if self.noise_re.search(sentence.text):
                    continue

                text = document["text_cleaned"][sentence.start_char:sentence.end_char]
                document["entities"][Entity.FOOTNOTE].add(text)

                self.clean_text(document, text)

        return document

    def get_sentences(self, document):
        try:
            result = self.sentencizer(document["text_cleaned"])
            return result.sents
        except KeyError:
            logging.error(f"cannot parse sentences of '{document['name']}'")
            return []
