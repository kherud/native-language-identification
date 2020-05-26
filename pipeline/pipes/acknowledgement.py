import re
from collections import defaultdict
from . import Target


class AcknowledgementParser(Target):
    def __init__(self):
        super().__init__()
        self.reference_re = re.compile(r"[0-9;,\[\]()]|et.? al.?|and")
        self.year_re = re.compile(r"[0-9]{4}")
        self.ack_re = re.compile("\s*".join("Acknowledgement") + "(?:\s*s)?\s*")


    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to acknowledgement parser"

        ack_mentions = list(self.ack_re.finditer(document["text"]))

        if len(ack_mentions) == 0:
            return document

        ack_mention = ack_mentions[-1]
        ack_text = document["text"][ack_mention.end():]

        # find authors' names in references
        reference_names = set()
        for reference in document["entities"]["reference"]:
            reference = self.reference_re.sub(" ", reference)
            reference = self.year_re.sub(" ", reference)
            for x in reference.split(" "):
                if len(x) < 2 or not x.isalpha():
                    continue
                reference_names.add(x)

        ack_end = "R\s?eference|Conclusion|Discussion|Case\sStudies|Results|Related\sWork|Appendix|Proof|Theorem|Table|Figure" + "|".join(reference_names)

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

        document["entities"]["acknowledgement"].add(ack)

        return document


