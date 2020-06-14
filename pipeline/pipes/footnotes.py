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
        self.keyword_re = re.compile(r"acknowledges?\s(?:the\s)?support|was\s(?:\w{1,10}\s)?supported\sby")

        self.sentencizer_dir = os.path.abspath(sentencizer_dir)
        assert os.path.exists(self.sentencizer_dir), f"ner model directory '{self.sentencizer_dir}' does not exist"
        self.sentencizer = spacy.load(self.sentencizer_dir)

        self.abstract_re = re.compile("\s*".join("ABSTRACT") + "|" + "\s*".join("Abstract"))

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

        sentences = self.get_sentences(document)
        for sentence in sentences:
            if any(re.search(r"[\s\-]*".join(re.escape(k) for k in key_word.split()), sentence.text, re.IGNORECASE) for key_word in key_words) \
                    or self.keyword_re.search(sentence.text):
                if self.noise_re.search(sentence.text):
                    continue
                text = document["text_cleaned"][sentence.start_char:sentence.end_char]
                document["entities"][Entity.FOOTNOTE].add(text)

        for footnote in document["entities"][Entity.FOOTNOTE]:
            self.clean_text(document, footnote)

        return document

    def get_sentences(self, document):
        try:
            abstract_start, abstract_end = self.detect_abstract(document)
            return self.sentencizer(document["text_cleaned"][abstract_end:]).sents
        except KeyError:
            logging.error(f"cannot parse sentences of '{document['name']}'")
            return []

    def detect_abstract(self, document):
        abstract_mention = None
        abstract_title_mentions = []
        if "meta" in document:
            abstract_title_mentions = self.abstract_re.findall(document["meta"]["title"])
        if len(abstract_title_mentions) > 0:
            abstract_mentions = list(self.abstract_re.finditer(document["text_cleaned"]))
            if len(abstract_mentions) >= len(abstract_title_mentions):
                abstract_mention = abstract_mentions[len(abstract_title_mentions)]
        else:
            abstract_mention = self.abstract_re.search(document["text_cleaned"])

        if abstract_mention is None:
            return 400, 400
        else:
            start = min(abstract_mention.start(), 1000)
            end = min(abstract_mention.end(), 1000)
            return start, end
