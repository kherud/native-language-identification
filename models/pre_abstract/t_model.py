import json
import glob
import torch
import random
from tokenizers import ByteLevelBPETokenizer


# models = []
# for tokenizer_path in glob.glob("tokenizer/*/*"):
#     model = torch.load(tokenizer_path.replace("tokenizer", "checkpoints"))
#     with open(f"{tokenizer_path}/vocab.json", "r") as file:
#         vocab = json.load(file)
#         if any(x.isupper() for x in vocab.keys()):
#             cased = True
#         else:
#             cased = False
#     tokenizer = ByteLevelBPETokenizer(vocab_file=f"{tokenizer_path}/vocab.json",
#                                       merges_file=f"{tokenizer_path}/merges.txt",
#                                       lowercase=cased)
#
#     print(model_path, tokenizer_path)
#
model = torch.load("model/lstm-tagger.pt")
tokenizer = ByteLevelBPETokenizer(vocab_file="model/vocab.json",
                                  merges_file="model/merges.txt",
                                  lowercase=False)
with open("model/classes.json", "r") as file:
    classes = json.load(file)
    classes = {v: k for k, v in classes.items()}

file_paths = [random.choice(glob.glob("../../data/pre_abstract_txts/*.txt")) for _ in range(25)]
for index, file_path in enumerate(file_paths):
    with open(file_path, "r") as file:
        text = file.read()

    with open(f"samples/sample{index}.txt", "w") as file:
        for line, prediction in model.annotate(classes, tokenizer, text):
            file.write(prediction + ", " + line + "\n")