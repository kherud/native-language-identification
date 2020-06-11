import os
import copy
import glob
import json
import tqdm
import torch
import random
import itertools
import multiprocessing
import numpy as np
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer


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


class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim, n_classes, dropout):
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.n_classes = n_classes

        self.dropout = nn.Dropout(p=dropout)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.td_lstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True)

        self.alignment = nn.Linear(lstm_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.lstm = nn.LSTM(lstm_dim, lstm_dim // 4, batch_first=True, bidirectional=True)

        self.classifier = nn.LSTM(lstm_dim // 2, self.n_classes, batch_first=True)

    def forward(self, x):
        batch, sentences, words = x.size()

        embeddings = self.embeddings(x)
        embeddings = embeddings.view(batch * sentences, words, self.embedding_dim)
        # embeddings = self.dropout(embeddings)

        # output shape [batches * sentences, words, lstm_dim]
        td_lstm_out, _ = self.td_lstm(embeddings)

        # calculate attention weighted state vector
        td_lstm_out = td_lstm_out.reshape(batch * sentences * words, self.lstm_dim)
        alignment = self.tanh(self.alignment(td_lstm_out))
        alignment = alignment.view(batch * sentences, words)
        attention = self.softmax(alignment)
        td_lstm_out = td_lstm_out.reshape(batch * sentences, words, self.lstm_dim)
        # [sentences,atttention_per_word] x [sentences,words,lstm_dim] -> [sentences,lstm_dim]
        attention_vectors = torch.einsum("sa,sal->sl", attention, td_lstm_out)
        attention_vectors = attention_vectors.view(batch, sentences, self.lstm_dim)

        # get last hidden state alternatively
        # td_lstm_out = td_lstm_out[:, -1, :]
        # td_lstm_out = td.lstm_out.view(batch, sentences, self.lstm_dim)

        lstm_out, _ = self.lstm(attention_vectors)
        # lstm_out = self.dropout(lstm_out)

        classifier_out, _ = self.classifier(lstm_out)

        return classifier_out

    def annotate(self, text):
        lines = text.split("\n")

        tokenized = [x.ids for x in self.tokenizer.encode_batch(lines)]

        # padding
        max_tokens = max(len(sentence) for sentence in tokenized)
        for sentence in range(len(tokenized)):
            for _ in range(max_tokens - len(tokenized[sentence])):
                tokenized[sentence].insert(0, 0)

        predictions = self.forward(torch.tensor([tokenized]))
        predictions = torch.argmax(predictions[0], -1)
        predictions = [self.classes[prediction.item()] for prediction in predictions]

        return zip(lines, predictions)


def architecture_search(process_id):
    os.makedirs(f"checkpoints/{process_id+1}")
    os.makedirs(f"tokenizer/{process_id+1}")

    files = glob.glob("../../data/pre_abstract_txts/*.txt")

    tok_sizes = list(range(100, 2000, 100))
    hidden_sizes = list(range(12, 300, 12))
    emb_sizes = list(range(10, 250, 10))
    cased = [True, False]

    # for v in np.geomspace(1e-2, 1, 5):
    batch_size = 1

    results = {}
    choices = list(itertools.product(tok_sizes, hidden_sizes, emb_sizes, cased))
    random.shuffle(choices)

    best_acc = -np.inf
    while len(choices) > 0:
        tok_size, hidden_size, emb_size, cased = choices.pop()
        print(tok_size, hidden_size, emb_size, cased)

        tokenizer = ByteLevelBPETokenizer(lowercase=cased)
        tokenizer.train(files, vocab_size=tok_size, special_tokens=["[PAD]"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = TextDataset(data_dir="../../data/pre_abstract_txts",
                              labels_dir="../../data/pre_abstract_labels",
                              device=device,
                              tokenizer=tokenizer,
                              batch_size=batch_size)
        test_dataset = TextDataset(data_dir="../../data/pre_abstract_txts",
                                   labels_dir="../../data/pre_abstract_labels_test",
                                   device=device,
                                   tokenizer=tokenizer,
                                   batch_size=batch_size)
        model = LSTMTagger(vocab_size=tokenizer.get_vocab_size(),
                           embedding_dim=emb_size,
                           lstm_dim=hidden_size,
                           dropout=0,
                           n_classes=len(dataset.classes)).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=v)

        epoch = 0
        n = 3
        test_acc = -np.inf
        log_interval = 10  # all n batches
        weights = copy.deepcopy(model.state_dict())
        while True:
            dataset.shuffle()
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
                acc = c / a
                if acc <= test_acc and n > 0:
                    n -= 1
                    continue
                elif acc <= test_acc:
                    break
                print(t_loss, acc)
                weights = copy.deepcopy(model.state_dict())
                test_acc = acc
        results[(tok_size, hidden_size, emb_size, cased)] = acc
        print(list(sorted([(k, v) for k, v in results.items()], key=lambda y: y[1], reverse=True))[:10])
        print(best_acc, test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            dir_path = f"tokenizer/{process_id+1}/lstm-tagger-{best_acc:.6f}"
            if os.path.exists(dir_path):
                continue
            torch.save(weights, f"checkpoints/{process_id+1}/lstm-tagger-{best_acc:.6f}.pt")
            os.makedirs(dir_path)
            tokenizer.save(dir_path)


"""
uncased: [((1200, 216, 120), 0.9604863221884499), ((500, 132, 210), 0.9574468085106383), ((800, 144, 160), 0.9544072948328267), ((1700, 96, 240), 0.9544072948328267), ((200, 168, 140), 0.9513677811550152), ((300, 192, 230), 0.9513677811550152), ((300, 204, 220), 0.9483282674772037), ((1000, 156, 130), 0.9483282674772037), ((1100, 108, 200), 0.9452887537993921), ((100, 192, 170), 0.9452887537993921)]
cased: [((700, 228, 230), 0.9696048632218845), ((200, 192, 130), 0.9635258358662614), ((400, 192, 220), 0.9635258358662614), ((1100, 192, 150), 0.9604863221884499), ((900, 228, 210), 0.9604863221884499), ((100, 204, 180), 0.9574468085106383), ((100, 156, 100), 0.9574468085106383), ((600, 156, 210), 0.9544072948328267), ((100, 204, 230), 0.9544072948328267), ((400, 204, 170), 0.9544072948328267)]
"""


if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        pool.map(architecture_search, range(4))


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
