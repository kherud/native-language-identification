import re
import json
import torch
import logging
from . import Target, Entity
from itertools import combinations
from os.path import abspath, exists, join
from tokenizers import ByteLevelBPETokenizer
from models.references.model import LSTMTagger


class ReferenceParser(Target):
    def __init__(self, model_dir, device: str = "cpu"):
        super().__init__()
        self.model_dir = abspath(model_dir)

        assert exists(self.model_dir), f"model directory '{self.model_dir}' does not exist"
        assert exists(join(self.model_dir, "config.json")), f"configuration file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "merges.txt")), f"merges file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "weights.pt")), f"model file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "vocab.json")), f"vocab file does not exist in {self.model_dir}"

        with open(join(self.model_dir, "config.json"), "r") as config_file:
            self.model_config = json.load(config_file)
        if not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = LSTMTagger(vocab_size=self.model_config["vocab_size"],
                                embedding_dim=self.model_config["embedding_dim"],
                                lstm_dim=self.model_config["lstm_dim"],
                                device=device).to(self.device)
        weights = torch.load(join(self.model_dir, "weights.pt"), map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()
        self.tokenizer = ByteLevelBPETokenizer(vocab_file=join(self.model_dir, "vocab.json"),
                                               merges_file=join(self.model_dir, "merges.txt"),
                                               lowercase=self.model_config["lowercase"])

        # self.space_re = re.compile(r"\s+")
        self.refer_re = re.compile(r"([(\[](\w[^\d()\[\]]+\s[(\[]?[12][0-9]{3}[A-Za-z]?[(\[]?[;,\s]*)+[)\]])")
        self.et_al_re = re.compile(r"[^\d\s]+\set\.?\s*al\.?,?\s+([(\[][12][0-9]{3}[)\]])?")
        self.and_re = re.compile(r"([^\d\s]+,?\s+((and)|&)\s+)+[^\d\s]+[,\s]*([(\[][12][0-9]{3}[A-Za-z]?[)\]]|[12][0-9]{3}[A-Za-z]?)")
        self.name_re = re.compile(r"[A-Z][^A-Z\d\s()\[\],;\.]+")

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to reference extractor"

        try:
            self.find_reference_blocks(document)
        except RuntimeError:
            logging.error(f"could not parse reference blocks of {document['name']}")

        text_references = self.find_text_references(document)

        # remove references from text, sort to match longest first
        sorted_ref = sorted(text_references, key=lambda ref: len(ref[0]), reverse=True)
        for reference in sorted_ref:
            self.clean_text(document, reference[0])

        return document

    def find_reference_blocks(self, document):
        lines = document["text_cleaned"].split("\n")
        labels = []

        # chunk text and prepare for model
        for i in range(0, len(lines), self.model_config["chunk_size"]):
            chunk = [x.ids for x in self.tokenizer.encode_batch(lines[i:i + self.model_config["chunk_size"]])]
            # padding
            max_line = max(len(line) for line in chunk)
            for j in range(len(chunk)):
                chunk[j] = [0] * (max_line - len(chunk[j])) + chunk[j]
            chunk = torch.Tensor([chunk]).long().to(self.device)

            predictions = self.model.forward(chunk, i, len(lines))
            labels.extend(torch.argmax(predictions[0], -1))

        keep_lines = []
        last_label, last_block = 0, []
        for line, label in zip(lines, labels):
            if last_label == 1 and label == 0 and len(last_block) > 0:
                document["entities"][Entity.REFERENCE].add("\n".join(last_block))
                last_block = []
            if label == 0:
                keep_lines.append(line)
            else:
                last_block.append(line)
            last_label = label
        if len(last_block) > 0:
            document["entities"][Entity.REFERENCE].add("\n".join(last_block))

        document["text_cleaned"] = "\n".join(keep_lines)

    def find_text_references(self, document):
        # find references based on different regular expressions
        references = []
        for x in self.refer_re.finditer(document["text_cleaned"]):
            start, end = x.span()
            references.append((document["text_cleaned"][start:end], start, end))
        for x in self.et_al_re.finditer(document["text_cleaned"]):
            start, end = x.span()
            references.append((document["text_cleaned"][start:end], start, end))
        for x in self.and_re.finditer(document["text_cleaned"]):
            start, end = x.span()
            references.append((document["text_cleaned"][start:end], start, end))

        # remove overlapping references
        for refer_a, refer_b in combinations(references, 2):
            if refer_a[1] <= refer_b[1] < refer_a[2] and refer_b[2] <= refer_a[2]:
                references.remove(refer_a)
                references.remove(refer_b)
                start = min(refer_a[1], refer_b[1])
                end = max(refer_a[2], refer_b[2])
                references.append((document["text_cleaned"][start:end], start, end))

        # extract names of authors in references and add reference entity
        document["references_authors"] = set()
        for reference in references:
            for x in self.name_re.findall(reference[0]):
                document["references_authors"].add(x)
            document["entities"][Entity.REFERENCE].add(reference[0])

        return references


class DeprecatedReferenceParser(Target):
    def __init__(self):
        super().__init__()
        self.space_re = re.compile(r"\s+")
        self.refer_re = re.compile(r"([(\[](?:\w[^\d()\[\]]+\s[(\[]?[12][0-9]{3}[A-Za-z]?[(\[]?[;,\s]*)+[)\]])")
        self.et_al_re = re.compile(r"[^\d\s]+\set\.?\s*al\.?,?\s+(?:[(\[][12][0-9]{3}[)\]])?")
        self.and_re = re.compile(r"(?:[^\d\s]+,?\s+(?:(?:and)|&)\s+)+[^\d\s]+[,\s]*(?:[(\[][12][0-9]{3}[A-Za-z]?[)\]]|[12][0-9]{3}[A-Za-z]?)")
        self.name_re = re.compile(r"[A-Z][^A-Z\d\s()\[\],;\.]+")

        self.year_re = re.compile(r"19[0-9]{2}|200[0-9]|201[0-9]")
        self.numref_re = re.compile(r"\[[1-9]\d{0,2}\]")
        self.pageline_re = re.compile(r"^\d+$")

        refblock_re1 = "\s*".join("Reference") + "\s*s?\s*"
        refblock_re1 += "|" + "\s*".join("REFERENCE") + "\s*S?\s*"
        self.refblock_re1 = re.compile(refblock_re1)
        self.refblock_re2 = re.compile("\s*".join("Literatur") + "\s*e?\s*")
        self.refblock_re3 = re.compile("\s*".join("Bibliography"))

        self.block_tolerance = 4

        self.keywords = {'international', 'proceedings', 'conference', 'journal', 'aaai', 'arxiv', 'doi', 'preprint',
                         'springer', 'advances', 'icml', 'association', 'acm', 'ieee', 'nips', 'cvpr', 'press', 'acl',
                         'naacl', 'emnlp'}

    def find_text_references(self, document):
        # replace multiple whitespaces with single space
        text = self.space_re.sub(" ", document["text"])

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

        # extract names of authors in references and add reference entity
        document["references_authors"] = set()
        for reference in references:
            for x in self.name_re.findall(reference[0]):
                document["references_authors"].add(x)
            document["entities"][Entity.REFERENCE].add(reference[0])

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to reference extractor"

        self.find_text_references(document)
        self.find_reference_block(document)

        # remove references from text, sort to match longest first
        sorted_ref = sorted(document["entities"][Entity.REFERENCE], key=lambda ref: len(ref), reverse=True)
        for reference in sorted_ref:
            self.clean_text(document, reference)

        return document

    def find_text_references(self, document):
        # replace multiple whitespaces with single space
        text = self.space_re.sub(" ", document["text"])

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

        # extract names of authors in references and add reference entity
        document["references_authors"] = set()
        for reference in references:
            for x in self.name_re.findall(reference[0]):
                document["references_authors"].add(x)
            document["entities"][Entity.REFERENCE].add(reference[0])


    def find_reference_block(self, document):
        mention = None
        for mention in self.refblock_re1.finditer(document["text"]):
            pass  # iterate to last occurrence

        if mention is None:
            for mention in self.refblock_re2.finditer(document["text"]):
                pass  # iterate to last occurrence

        if mention is None:
            for mention in self.refblock_re3.finditer(document["text"]):
                pass  # iterate to last occurrence

        if mention is None:
            return

        lines = document["text_cleaned"][mention.end():].split("\n")
        names = document["references_authors"]

        while len(lines) > 0:
            reference_block, lines = self.lines_to_block(lines, names)
            if len(reference_block) > 0:
                document["entities"][Entity.REFERENCE].add(reference_block)
                self.clean_text(document, reference_block)
            while len(lines) > 0 and not (
                    any(re.search("[\s\-]*".join(re.escape(c) for c in name.split()), lines[0]) for name in names) or
                    self.numref_re.search(lines[0])
            ):
                lines.pop(0)

    def lines_to_block(self, lines, names):
        index, c = 0, 0
        for index, line in enumerate(lines):
            if c == self.block_tolerance:
                break

            # search for any clue that the line belongs to references
            if len(self.year_re.findall(line)) > 0 \
                    or any(re.search("\s*".join(keyword), line, re.IGNORECASE) for keyword in self.keywords) \
                    or self.numref_re.search(line) \
                    or any(re.search(r"[\s\-]*".join(re.escape(c) for c in name.split()), line) for name in names) \
                    or self.pageline_re.match(line):
                c = 0
            else:
                c += 1

        # join lines to single string and return remaining lines
        return "\n".join(lines[:index - c]), lines[index:]
