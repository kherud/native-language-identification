import os
import glob
import json
import tqdm
import torch
import random
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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

            x = torch.tensor(batch[0]).to(self.device)
            y = torch.tensor(batch[1]).to(self.device)
            self.batches.append((x, y))


    def __getitem__(self, i):
        x, y = self.batches[i]
        return x, y

    def __len__(self):
        return len(self.batches)


class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, lstm_dim, n_classes, dropout):
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.lstm_dim = lstm_dim
        self.n_classes = n_classes
        self.Ks = [2, 3, 4, 6]

        self.dropout = nn.Dropout(p=dropout)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([nn.Conv2d(1, n_filters, (K, embedding_dim)) for K in self.Ks])

        self.lstm = nn.LSTM(4 * n_filters, lstm_dim // 2, batch_first=True, bidirectional=True)

        self.classifier = nn.LSTM(lstm_dim, self.n_classes, batch_first=True)

    def forward(self, x):
        batch, sentences, words = x.size()

        embeddings = self.embeddings(x)
        embeddings = embeddings.view(batch * sentences, words, self.embedding_dim)
        # embeddings = self.dropout(embeddings)

        embeddings = embeddings.unsqueeze(1)  # (N, Ci, W, D)
        cnn_out = [F.relu(conv(embeddings)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        cnn_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_out]  # [(N, Co), ...]*len(Ks)
        cnn_out = torch.cat(cnn_out, 1).view(batch, sentences, self.n_filters * 4)

        lstm_out, _ = self.lstm(cnn_out)
        # lstm_out = self.dropout(lstm_out)

        classifier_out, _ = self.classifier(lstm_out)

        return classifier_out

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


"""
vocab_size = [(600, 17.400037109851837), (3500, 17.950848817825317), (1400, 17.96711540222168), (2300, 18.002858579158783), (1600, 18.044274270534515), (4400, 18.212088525295258), (1500, 18.278275430202484), (1700, 18.302800476551056), (900, 18.39490497112274), (800, 18.577040255069733), (700, 18.59476888179779), (500, 18.63107419013977), (2700, 18.779658615589142), (4500, 18.943014085292816), (2900, 19.00313800573349), (3300, 19.11995130777359), (1900, 19.121903896331787), (3400, 19.339268624782562), (2400, 19.41307646036148), (4000, 19.428328275680542), (2200, 19.46500450372696), (2100, 19.50793981552124), (2500, 19.531544983386993), (1000, 19.55955868959427), (4800, 19.569620966911316), (2600, 19.61939573287964), (3200, 19.622986137866974), (1300, 19.732526659965515), (4100, 19.74906075000763), (4200, 19.756294667720795), (1800, 19.90055137872696), (4900, 20.09227740764618), (3600, 20.09765750169754), (3800, 20.209920704364777), (2000, 20.302987456321716), (4700, 20.34429508447647), (4600, 20.450044214725494), (3900, 20.53396439552307), (1200, 20.556682288646698), (2800, 20.586784839630127), (4300, 20.92519724369049), (3700, 21.40639591217041), (3100, 21.5532243847847), (1100, 21.923060715198517), (3000, 22.764125108718872), (5000, 22.844439804553986)]
emb_dim = [(110, 17.72357475757599), (180, 17.983745098114014), (120, 18.00517302751541), (140, 18.11272406578064), (160, 18.257606208324432), (60, 18.335835337638855), (170, 18.871184706687927), (200, 19.003950893878937), (80, 19.113344311714172), (90, 19.32656717300415), (190, 19.47134518623352), (150, 19.608049511909485), (70, 20.425654709339142), (130, 20.740531086921692), (40, 20.93796706199646), (50, 21.037606596946716), (10, 21.075260937213898), (100, 21.2917263507843), (20, 22.295424103736877), (30, 22.40872836112976), (320, 18.576534181833267), (290, 19.747096061706543), (220, 19.772528290748596), (340, 19.931555688381195), (330, 20.00263160467148), (310, 20.031763017177582), (270, 20.300805389881134), (300, 20.448286771774292), (230, 20.479256004095078), (240, 20.48120814561844), (250, 20.53259539604187), (280, 21.105005860328674), (210, 22.695391476154327), (260, 23.075069963932037)]
dropout = {0.0: 19.48966360092163, 0.025: 18.129290521144867, 0.05: 19.973151445388794, 0.07500000000000001: 18.278468042612076, 0.1: 20.188306272029877, 0.125: 19.94023472070694, 0.2: 19.233620822429657, 0.3: 18.361724257469177, 0.4: 18.864712566137314, 0.5: 19.466215193271637, 0.6: 19.736424028873444, 0.7: 23.23954665660858}
lstm_dim = [(132, 17.192459225654602), (84, 17.575207352638245), (72, 17.719788134098053), (36, 18.150880306959152), (60, 18.495288014411926), (12, 18.572497189044952), (96, 18.99687957763672), (120, 19.00813454389572), (216, 19.097515404224396), (192, 19.757775157690048), (156, 20.29127848148346), (168, 20.311414301395416), (24, 20.333223164081573), (144, 20.361957669258118), (240, 20.690076768398285), (108, 21.159685909748077), (228, 21.173621356487274), (48, 21.91128009557724), (180, 22.178074717521667), (204, 22.725580751895905)]
"""


if __name__ == "__main__":
    losses = {}
    files = glob.glob("../../data/pre_abstract_txts/*.txt")

    tok_sizes = list(range(100, 2000, 100))
    hidden_sizes = list(range(12, 240, 12))
    emb_sizes = list(range(10, 100, 10))
    n_filters = list(range(5, 55, 5))

    # for v in np.geomspace(1e-2, 1, 5):
    epochs = 25
    batch_size = 1

    results = {}
    choices = list(itertools.product(tok_sizes, hidden_sizes, emb_sizes, n_filters))
    random.shuffle(choices)

    while len(choices) > 0:
        tok_size, hidden_size, emb_size, n_filters = choices.pop()
        print(tok_size, hidden_size, emb_size, n_filters)

        tokenizer = ByteLevelBPETokenizer(lowercase=True)
        tokenizer.train(files, vocab_size=tok_size, special_tokens=["[PAD]"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = TextDataset(data_dir="../../data/pre_abstract_txts",
                              labels_dir="../../data/pre_abstract_labels_old",
                              device=device,
                              tokenizer=tokenizer,
                              batch_size=batch_size)
        test_dataset = TextDataset(data_dir="../../data/pre_abstract_txts",
                                   labels_dir="../../data/pre_abstract_labels_test_i",
                                   device=device,
                                   tokenizer=tokenizer,
                                   batch_size=batch_size)
        model = LSTMTagger(vocab_size=tokenizer.get_vocab_size(),
                           embedding_dim=emb_size,
                           n_filters=n_filters,
                           lstm_dim=hidden_size,
                           dropout=0,
                           n_classes=len(dataset.classes)).to(device)  # .to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=v)

        epoch = 0
        n = 3
        test_acc = -np.inf
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

                pbar.set_description(f'epoch {epoch} | batch {i+1:d}/{len(dataset)} | loss {loss.item():.2f}')

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
                test_acc = acc
        results[(tok_size, hidden_size, emb_size, n_filters)] = acc
        print(list(sorted([(k,v) for k, v in results.items()], key=lambda y: y[1], reverse=True))[:10])
            # losses[v] = test_loss
            # print(losses)
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
        # torch.save(model.state_dict(), f"checkpoints/lstm-tagger-{epoch}.p")
    """
