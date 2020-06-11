import re
import json
import torch
import logging
from tokenizers import ByteLevelBPETokenizer
from os.path import exists, join, abspath
from . import Target, Entity


class PreAbstractParser(Target):
    def __init__(self, model_dir, device="cpu"):
        super().__init__()
        self.model_dir = abspath(model_dir)

        assert exists(self.model_dir), f"model directory '{self.model_dir}' does not exist"
        assert exists(join(self.model_dir, "classes.json")), f"classes file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "merges.txt")), f"merges file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "model.pt")), f"model file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "vocab.json")), f"vocab file does not exist in {self.model_dir}"

        if not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = torch.load(join(self.model_dir, "model.pt")).to(self.device)
        self.model.eval()
        self.tokenizer = ByteLevelBPETokenizer(vocab_file=join(self.model_dir, "vocab.json"),
                                               merges_file=join(self.model_dir, "merges.txt"),
                                               lowercase=False)
        with open(join(self.model_dir, "classes.json"), "r") as classes_file:
            self.class_to_index = json.load(classes_file)
            self.index_to_class = {v: k for k, v in self.class_to_index.items()}

        self.department_re = re.compile("(?:,\s*)?[^,]*Department[^,]*(?:,)", re.IGNORECASE)

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to author parser"

        lines, labels = self.annotate_lines(document["text"][:document["abstract_start"]])

        keep_lines = []
        for line, label in zip(lines, labels):
            if label == "other":
                keep_lines.append(line)
            else:
                self.create_annotation(document, line, label)

        if "meta" in document:
            keep_lines = self.post_process_lines(document, keep_lines)

        document["text_cleaned"] = "\n".join(keep_lines) + document["text"][document["abstract_start"]:]

        return document

    def annotate_lines(self, text):
        lines = text.split("\n")

        tokenized = [x.ids for x in self.tokenizer.encode_batch(lines)]

        # padding
        max_tokens = max(len(sentence) for sentence in tokenized)
        for sentence in range(len(tokenized)):
            for _ in range(max_tokens - len(tokenized[sentence])):
                tokenized[sentence].insert(0, 0)

        tensor = torch.tensor([tokenized]).to(self.device)
        predictions = self.model.forward(tensor)
        predictions = torch.argmax(predictions[0], -1)
        predictions = [self.index_to_class[prediction.item()] for prediction in predictions]

        return lines, predictions

    def create_annotation(self, document, line, label):
        if label == "private":
            document["entities"][Entity.PERSONAL_DATA].add(line)
        elif label == "author":
            document["entities"][Entity.AUTHOR].add(line)
        elif label == "email":
            document["entities"][Entity.EMAIL].add(line)
        elif label == "organization":
            for department_mention in self.department_re.findall(line):
                document["entities"][Entity.PERSONAL_DATA].add(department_mention)
            line = self.department_re.sub("", line)
            document["entities"][Entity.INSTITUTION_COMPANY].add(line)
        else:
            logging.error(f"label '{label}' not recognized in {type(self)}")
            raise ValueError(f"label '{label}' not recognized")

    def post_process_lines(self, document, lines):
        keep_lines = []
        for line in lines:
            mention = False

            try:
                for author in document["meta"]["authors"]:
                    mention = re.search("[\s\-]*".join(re.escape(name) for name in author.split()), line, re.IGNORECASE)
                if mention:
                    document["entities"][Entity.AUTHOR].add(line)
                    continue

                for organization in document["meta"]["orgs"]:
                    mention = re.search("[\s\-]*".join(re.escape(name) for name in organization["name"].split()), line, re.IGNORECASE)
                if mention:
                    document["entities"][Entity.INSTITUTION_COMPANY].add(line)
                    continue
            except KeyError:
                logging.error(f"conferences meta file misses key for {document['name']}")

            keep_lines.append(line)

        return keep_lines
