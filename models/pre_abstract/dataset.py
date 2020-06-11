import os
import glob
import json
import random
import torch


class TextDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_dir, labels_dir, device, tokenizer, batch_size=32):
        self.data_dir = data_dir
        self.device = device
        self.labels_dir = labels_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.batches = []
        self.classes = {}
        self.load_data()

    def load_data(self):
        data = []
        label_set = set()
        # load text files
        for file_path in glob.glob(f"{self.labels_dir}/*.txt"):
            file_name = file_path.split("/")[-1]
            with open(file_path, "r") as file:
                labels = file.readlines()
            try:
                with open(os.path.join(self.data_dir, file_name), "r") as file:
                    text = file.readlines()
            except FileNotFoundError:
                continue

            for label in labels:
                label_set.add(label.strip())

            n_lines = min(len(text), len(labels))
            data.append((text[:n_lines], labels[:n_lines]))

        self.classes = {label: index for index, label in enumerate(label_set)}
        with open("model/classes.json", "w") as file:
            json.dump(self.classes, file)

        # sort by amount lines to reduce padding
        data.sort(key=lambda x: len(x[0]))

        # create batches
        for i in range(len(data) // self.batch_size):
            batch = [[], []]
            for j in range(self.batch_size):
                batch[0].append([x.ids for x in self.tokenizer.encode_batch(data[i * self.batch_size + j][0])])
                batch[1].append([self.classes[x.strip()] for x in data[i * self.batch_size + j][1]])

                assert len(batch[0][-1]) == len(batch[1][-1]), "amount labels does not match data"

            max_sentences = max(len(sentences) for sentences in batch[0])
            max_tokens = max(len(tokens) for sentences in batch[0] for tokens in sentences)

            # padding
            for bi in range(len(batch[0])):
                # pad existing sentences
                for si in range(len(batch[0][bi])):
                    for _ in range(max_tokens - len(batch[0][bi][si])):
                        batch[0][bi][si].insert(0, 0)
                # append empty sentences
                batch[0][bi].extend([[0] * max_tokens] * (max_sentences - len(batch[0][bi])))
                batch[1][bi].extend([self.classes["other"]] * (max_sentences - len(batch[1][bi])))

            x = torch.tensor(batch[0]).to(self.device)
            y = torch.tensor(batch[1]).to(self.device)
            self.batches.append((x, y))

    def __getitem__(self, i):
        x, y = self.batches[i]
        return x, y

    def __len__(self):
        return len(self.batches)

    def shuffle(self):
        random.shuffle(self.batches)