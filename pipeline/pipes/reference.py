import re
import string
from . import Target, Entity
from itertools import combinations


class ReferenceParser(Target):
    def __init__(self):
        super().__init__()
        chars = "A-Za-z\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF\-"
        self.space_re = re.compile(r"\s+")
        self.noise_re = re.compile(f"[^{string.printable}\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF]")
        self.refer_re = re.compile(r"([(\[](?:\w[^\d()\[\]]+\s[(\[]?[0-9]{4}[A-Za-z]?[(\[]?[;,\s]*)+[)\]])")
        self.et_al_re = re.compile(rf"[{chars}]+\set\.?\s*al\.?\s+(?:[(\[][0-9]{{4}}[)\]])?")
        self.and_re = re.compile(rf"(?:[{chars}]+,?\s+(?:(?:and)|&)\s+)+[{chars}]+[,\s]*(?:[(\[][0-9]{{4}}[A-Za-z]?[)\]]|[0-9]{{4}}[A-Za-z]?)")
        self.name_re = re.compile(r"[0-9;,\[\]()]|et.? al.?|and")

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to reference extractor"

        text = self.space_re.sub(" ", document["text"])  # replace multiple whitespaces with single space
        text = self.noise_re.sub("", text)

        # find references based on different regular expressions
        references = []
        for x in self.refer_re.finditer(text):
            start, end = x.span()
            references.append((text[start:end], start, end))
        for x in self.et_al_re.finditer(text):
            start, end = x.span()
            references.append((text[start:end], start, end))
        for x in self.and_re.finditer(text):
            start, end = x.span()
            references.append((text[start:end], start, end))

        # remove overlapping references
        for refer_a, refer_b in combinations(references, 2):
            if refer_a[1] <= refer_b[1] < refer_a[2] and refer_b[2] <= refer_a[2]:
                references.remove(refer_a)
                references.remove(refer_b)
                start = min(refer_a[1], refer_b[1])
                end = max(refer_a[2], refer_b[2])
                references.append((text[start:end], start, end))

        # extract names of authors in references
        document["references_authors"] = set()
        for reference in references:
            reference = self.name_re.sub(" ", reference[0])
            reference = self.space_re.sub(" ", reference)
            for x in reference.split(" "):
                if len(x) < 2 or not x.isalpha():
                    continue
                document["references_authors"].add(x)

        for refer in references:
            document["entities"][Entity.REFERENCE].add(refer[0])

        return document