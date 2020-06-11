import os
import glob
import copy
import tqdm
import random
import itertools
import torch
import torch.nn as nn
import numpy as np
from multiprocessing import Pool
from tokenizers import ByteLevelBPETokenizer

from models.pre_abstract.dataset import TextDataset
from models.pre_abstract.model import LSTMTagger

"""
Quick messy code
"""

def architecture_search(process_id):
    os.makedirs(f"checkpoints/{process_id+1}")
    os.makedirs(f"tokenizer/{process_id+1}")

    files = glob.glob("../../data/pre_abstract_txts/*.txt")

    tok_sizes = list(range(100, 2000, 100))
    hidden_sizes = list(range(12, 300, 12))
    emb_sizes = list(range(10, 250, 10))
    cased = [True, False]

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


if __name__ == "__main__":
    # start multiple processes
    with Pool() as pool:
        pool.map(architecture_search, range(4))
