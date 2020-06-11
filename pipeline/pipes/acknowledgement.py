import re
from . import Target, Entity


class AcknowledgementParser(Target):
    def __init__(self):
        super().__init__()
        self.reference_re = re.compile(r"[0-9;,\[\]()]|et\.?\sal\.?|and")
        self.year_re = re.compile(r"[0-9]{4}")
        ack_re = "\s*".join("Acknowledg") + "e?" + "\s*".join("ment") + "(?:\s*s)?\s*"
        ack_re += "|" + "\s*".join("ACKNOWLEDG") + "E?" + "\s*".join("MENT") + "(?:\s*s)?\s*"
        self.ack_re = re.compile(ack_re)

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to acknowledgement parser"

        mention = None
        for mention in self.ack_re.finditer(document["text_cleaned"]):
            pass  # iterate to last occurrence

        if mention is None:
            return document

        ack_text = document["text_cleaned"][mention.end():]

        ack_end = "R\s?eference|C\s?onclusion|D\s?iscussion|Case\sStudies|Proposition|R\s?esults|Related\sWork|A\s?ppendix|Proof|Theorem|Table|Figure"
        if "references_authors" in document:
            ack_end += "|".join(re.escape(x) for x in document["references_authors"])

        ack = re.search(rf"(.*)(\n(\d{{1,3}})?\n+(.{{1,50}})?({ack_end}))", ack_text, re.IGNORECASE)

        if ack is not None:
            ack = ack_text[:ack.end()]
        elif len(ack_text) < 500:  # probably last element in paper
            ack = ack_text
        else:  # last try to find an ending
            ack = re.search(r".*\n(?:\d{1,3})?\n+", ack_text)
            ack = ack_text[:ack.end()]
        if ack is None:
            ack = ack_text

        ack = re.sub(rf"\n.?{ack_end}$", "", ack)

        document["entities"][Entity.ACKNOWLEDGEMENT].add(ack)

        self.clean_text(document, ack)

        return document


