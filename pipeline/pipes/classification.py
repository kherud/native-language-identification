import os
import json
import torch
import random
import numpy as np
from torch.cuda.amp import autocast
from os.path import exists, join
from models.classification.model import LSTM
from . import Target


class LSTMClassification(Target):
    def __init__(self,
                 model_dir: str,
                 device: str):
        super().__init__()
        self.model_dir = model_dir

        assert exists(join(model_dir, "labels.json")), f"labels file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "config.json")), f"configuration file does not exist in {self.model_dir}"
        assert exists(join(self.model_dir, "weights.pt")), f"weights file does not exist in {self.model_dir}"

        with open(join(model_dir, "labels.json"), "r") as file:
            self.labels = json.load(file)
            self.index2label = {v: k for k, v in self.labels.items()}
        with open(join(model_dir, "config.json"), "r") as file:
            self.model_config = json.load(file)

        if not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = LSTM(n_classes=len(self.labels),
                          vocab_size=self.model_config["vocab_size"],
                          embedding_dim=self.model_config["embedding_dim"],
                          hidden_dim=self.model_config["hidden_dim"]).to(self.device)
        weights = torch.load(join(self.model_dir, "weights.pt"), map_location=device)
        self.model.load_state_dict(weights)
        self.model = self.model.eval()

    def __call__(self, document):
        assert isinstance(document, dict), f"wrong input of type {type(document)} to classifier"

        in_tensor = torch.tensor([document["input_ids"]]).to(self.device)

        if self.device.type == "cpu":
            output = self.model(in_tensor)
        else:
            with autocast():
                output = self.model(in_tensor)

        prediction = torch.argmax(output, dim=-1).item()

        document["prediction"] = self.index2label[prediction]

        return document

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
