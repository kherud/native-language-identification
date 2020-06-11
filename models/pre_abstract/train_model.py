import os
import glob
import json
import tqdm
import torch
import random
import numpy as np
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer


class TextDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data_dir, labels_dir, tokenizer, batch_size=32):
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.batches = []
        self.classes = {}
        self.load_data()

    def load_data(self):
        data = []
        label_set = set()
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

        for i in range(len(data) // self.batch_size):
            batch = [[], []]
            for j in range(self.batch_size):
                batch[0].append([x.ids for x in self.tokenizer.encode_batch(data[i * self.batch_size + j][0])])
                batch[1].append([self.classes[x.strip()] for x in data[i * self.batch_size + j][1]])

                assert len(batch[0][-1]) == len(batch[1][-1]), "amount labels does not match data"

            max_sentences = max(len(sentences) for sentences in batch[0])
            max_tokens = max(len(tokens) for sentences in batch[0] for tokens in sentences)

            for bi in range(len(batch[0])):
                # pad existing sentences
                for si in range(len(batch[0][bi])):
                    for _ in range(max_tokens - len(batch[0][bi][si])):
                        batch[0][bi][si].insert(0, 0)
                # append empty sentences
                batch[0][bi].extend([[0] * max_tokens] * (max_sentences - len(batch[0][bi])))
                batch[1][bi].extend([self.classes["other"]] * (max_sentences - len(batch[1][bi])))

            self.batches.append(batch)


    def __getitem__(self, i):
        x, y = self.batches[i][0], self.batches[i][1]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.batches)


class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim, n_classes, dropout):
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.n_classes = n_classes

        self.dropout = nn.Dropout(p=dropout)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.td_lstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True)
        self.lstm = nn.LSTM(lstm_dim, lstm_dim // 4, batch_first=True, bidirectional=True)

        self.classifier = nn.LSTM(lstm_dim // 2, self.n_classes, batch_first=True)

    def forward(self, x):
        batch, sentences, words = x.size()

        embeddings = self.embeddings(x)
        embeddings = embeddings.view(batch * sentences, words, self.embedding_dim)
        # embeddings = self.dropout(embeddings)

        td_lstm_out, _ = self.td_lstm(embeddings)
        # get last hidden state
        td_lstm_out = td_lstm_out[:, -1, :].view(batch, sentences, self.lstm_dim)
        # td_lstm_out = self.dropout(td_lstm_out)

        lstm_out, _ = self.lstm(td_lstm_out)
        # lstm_out = self.dropout(lstm_out)

        classifier_out, _ = self.classifier(lstm_out)

        return classifier_out


"""
vocab_size: [(2500, 55.06892216205597), (1800, 55.296554535627365), (1700, 55.70338034629822), (2400, 55.919392704963684), (600, 56.623719304800034), (2600, 56.80652382969856), (1900, 56.83736073970795), (2100, 57.277023017406464), (800, 57.429887145757675), (2700, 58.15184444189072), (1300, 58.515746027231216), (700, 58.724223017692566), (500, 58.880771934986115), (1000, 59.9957021176815), (2800, 60.22263592481613), (1500, 60.320818454027176), (2000, 60.3958480656147), (2900, 61.12214756011963), (900, 61.85780391097069), (2300, 62.74037793278694), (1600, 63.128984212875366), (3000, 67.86270919442177), (1200, 68.26325562596321), (1100, 71.09749701619148), (1400, 71.2525565624237), (2200, 76.6799276471138)]
"""


if __name__ == "__main__":
    losses = {}
    files = glob.glob("../../data/pre_abstract_txts/*.txt")
    for v in range(5, 51):
        epochs = 25
        batch_size = 1

        tokenizer = ByteLevelBPETokenizer(lowercase=True)
        tokenizer.train(files, vocab_size=v * 100, special_tokens=["[PAD]"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = TextDataset(data_dir="../../data/pre_abstract_txts",
                              labels_dir="../../data/pre_abstract_labels_old",
                              tokenizer=tokenizer,
                              batch_size=batch_size)
        test_dataset = TextDataset(data_dir="../../data/pre_abstract_txts",
                                   labels_dir="../../data/pre_abstract_labels_test_i",
                                   tokenizer=tokenizer,
                                   batch_size=batch_size)
        model = LSTMTagger(vocab_size=tokenizer.get_vocab_size(),
                           embedding_dim=50,
                           lstm_dim=20,
                           dropout=0.1,
                           n_classes=len(dataset.classes))  # .to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        epoch = 0
        n = 5
        test_loss = -np.inf
        log_interval = 10  # all n batches
        while True:
            epoch += 1
            model.train()
            total_loss = 0.
            pbar = tqdm.tqdm(enumerate(dataset), desc=f"epoch {epoch}")
            for i, (x, y) in pbar:
                # reset gradients
                optimizer.zero_grad()
                # feed forward batch
                output = model(x)
                # calculate loss
                loss = criterion(output.transpose(1, 2), y)
                # back propagate loss
                loss.backward()
                # norm and clip gradients
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                pbar.set_description(f'epoch {epoch} | batch {i + 1:d}/{len(dataset)} | loss {loss.item():.2f}')

            model.eval()
            a, c = 0, 0
            with torch.no_grad():
                t_loss = 0
                for i, (x, y) in enumerate(test_dataset):
                    output = model(x)
                    loss = criterion(output.transpose(1, 2), y)
                    t_loss += loss.item()
                    for p, t in zip(torch.argmax(output, -1), y):
                        for pi, ti in zip(p, t):
                            a += 1
                            if pi == ti:
                                c += 1
                if c / a < test_loss and n > 0:
                    n -= 1
                    continue
                elif c / a < test_loss:
                    break
                print(t_loss, c / a)
                test_loss = c / a
        losses[v * 100] = test_loss
        print(losses)
    """
    classes = {v: k for k, v in dataset.classes.items()}

    file_paths = [random.choice(glob.glob("../../data/pre_abstract_txts/*.txt")) for _ in range(25)]
    for index, file_path in enumerate(file_paths):
        with open(file_path, "r") as file:
            lines = file.readlines()
            batch = [x.ids for x in tokenizer.encode_batch(lines)]

        max_tokens = max(len(sentence) for sentence in batch)

        for si in range(len(batch)):
            for _ in range(max_tokens - len(batch[si])):
                batch[si].insert(0, 0)

        output = model(torch.tensor([batch]))
        with open(f"samples/sample{index}.txt", "w") as file:
            for line, prediction in zip(lines, torch.argmax(output[0], -1)):
                file.write(classes[prediction.item()] + ", " + line)
    """
        # torch.save(model.state_dict(), f"checkpoints/lstm-tagger-{epoch}.p")