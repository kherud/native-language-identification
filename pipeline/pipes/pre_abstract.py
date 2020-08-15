import re
import json
import torch
import logging
from tokenizers import ByteLevelBPETokenizer
from os.path import exists, join, abspath
from . import Target, Entity
from models.pre_abstract.model import LSTMTagger


class PreAbstractParser(Target):
    def __init__(self, model_dir, device="cpu"):
        super().__init__()
        self.model_dir = abspath(model_dir)

        assert exists(self.model_dir), f"model directory '{self.model_dir}' does not exist"
        assert exists(join(self.model_dir, "classes.json")), f"classes file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "config.json")), f"configuration file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "merges.txt")), f"merges file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "weights.pt")), f"weights file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "vocab.json")), f"vocab file does not exist in {self.model_dir}"

        with open(join(self.model_dir, "classes.json"), "r") as classes_file:
            self.class_to_index = json.load(classes_file)
            self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        with open(join(self.model_dir, "config.json"), "r") as config_file:
            self.model_config = json.load(config_file)
        if not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = LSTMTagger(vocab_size=self.model_config["vocab_size"],
                                embedding_dim=self.model_config["embedding_dim"],
                                lstm_dim=self.model_config["lstm_dim"],
                                n_classes=len(self.class_to_index)).to(self.device)
        weights = torch.load(join(self.model_dir, "weights.pt"), map_location=device)
        self.model.load_state_dict(weights)
        self.model = self.model.eval()
        self.tokenizer = ByteLevelBPETokenizer(vocab_file=join(self.model_dir, "vocab.json"),
                                               merges_file=join(self.model_dir, "merges.txt"),
                                               lowercase=self.model_config["lowercase"])

        self.noise_re = re.compile(r"[^A-Za-z ]")

        self.department_re = re.compile(r"(?:,\s*)?[^,]*Department[^,]*(?:,)", re.IGNORECASE)

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to author parser"

        try:
            lines, labels = self.annotate_lines(document["text"][:document["abstract_start"]])
        except RuntimeError:
            logging.error(f"could not parse pre abstract of {document['name']}")
            return document

        keep_lines = []
        for line, label in zip(lines, labels):
            if "meta" in document and self.noise_re.sub("", line) == self.noise_re.sub("", document["meta"]["title"]):
                keep_lines.append(line)
            elif label == "other":
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
                    if re.search("[\s\-]*".join(re.escape(name) for name in author.split()), line, re.IGNORECASE):
                        mention = True
                        document["entities"][Entity.AUTHOR].add(line)

                for organization in document["meta"]["orgs"]:
                    if re.search("[\s\-]*".join(re.escape(name) for name in organization["name"].split()), line, re.IGNORECASE):
                        mention = True
                        document["entities"][Entity.INSTITUTION_COMPANY].add(line)
            except KeyError:
                logging.error(f"conferences meta file misses key for {document['name']}")

            if not mention:
                keep_lines.append(line)

        return keep_lines
