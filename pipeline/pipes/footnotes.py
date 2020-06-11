import re
from . import Target, Entity


class FootnoteParser(Target):
    def __init__(self):
        super().__init__()
        self.key_words = ["sponsored", "funded", "funding", "internship"]
        noise_re = "\s*".join("Acknowledg") + "e?" + "\s*".join("ment") + "(?:\s*s)?\s*"
        noise_re += "|" + "\s*".join("ACKNOWLEDG") + "E?" + "\s*".join("MENT") + "(?:\s*S)?\s*"
        noise_re += "|" + "\s*".join("Reference") + "\s*s?\s*"
        noise_re += "|" + "\s*".join("REFERENCE") + "\s*S?\s*"
        noise_re += "|" + "\[\d+\]"
        self.noise_re = re.compile(noise_re)
        self.keyword_re = re.compile(r"acknowledges?\s(?:the\s)?support|was\s(?:\w{1,10}\s)?supported\sby")

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to location parser"

        key_words = self.key_words
        key_words += list(document["entities"][Entity.AUTHOR])
        # only consider long keywords, since most short ones result in poor precision, e.g. google
        key_words += list(filter(lambda x: len(x) > 15,  document["entities"][Entity.INSTITUTION_COMPANY]))

        for sentence in document["sentences"]:
            if any(re.search("\s*".join(re.escape(c) for c in key_word), sentence.text, re.IGNORECASE) for key_word in key_words) \
                    or self.keyword_re.search(sentence.text):
                if self.noise_re.search(sentence.text):
                    continue
                document["entities"][Entity.FOOTNOTE].add(sentence.text)

                self.clean_text(document, sentence.text)

        return document
