import re
import string
from collections import defaultdict
from . import Target


class ReferenceParser(Target):
    def __init__(self):
        super().__init__()
        chars = "A-Za-z\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF\-"
        self.space_re = re.compile(r"\s+")
        self.noise_re = re.compile(f"[^{string.printable}\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF]")
        self.refer_re = re.compile(r"([(\[](?:\w[^\d()\[\]]+\s[(\[]?[0-9]{4}[A-Za-z]?[(\[]?[;,\s]*)+[)\]])")
        self.et_al_re = re.compile(rf"[{chars}]+\set\.?\s*al\.?\s+(?:[(\[][0-9]{{4}}[)\]])?")
        self.and_re = re.compile(rf"(?:[{chars}]+,?\s+(?:(?:and)|&)\s+)+[{chars}]+[,\s]*(?:[(\[][0-9]{{4}}[A-Za-z]?[)\]]|[0-9]{{4}}[A-Za-z]?)")

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to reference extractor"
        assert "text" in document, "no text in document"

        if "entities" not in document:
            document["entities"] = defaultdict(set)

        text = self.space_re.sub(" ", document["text"])  # replace multiple whitespaces with single space
        text = self.noise_re.sub("", text)

        all_references = []
        for x in self.refer_re.finditer(text):
            start, end = x.span()
            all_references.append((text[start:end], start, end))
        for x in self.et_al_re.finditer(text):
            start, end = x.span()
            all_references.append((text[start:end], start, end))
        for x in self.and_re.finditer(text):
            start, end = x.span()
            all_references.append((text[start:end], start, end))

        # remove overlapping references
        for refer_a in all_references:
            for refer_b in all_references:
                if refer_a[1] <= refer_b[1] < refer_a[2] and refer_b[2] <= refer_a[2]:
                    start = min(refer_a[1], refer_b[1])
                    end = max(refer_a[2], refer_b[2])
                    document["entities"]["reference"].add(text[start:end])

        return document