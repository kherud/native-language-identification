import os
import glob
import random
from readchar import readchar

file_paths = glob.glob("../../data/pre_abstract_txts/*.txt")
file_paths = list(filter(lambda x: not os.path.exists(x.replace("abstract_txts", "abstract_labels")), file_paths))

random.shuffle(file_paths)

for file_path in file_paths:
    with open(file_path, "r") as file:
        text = file.read()

    document = []
    lines = text.splitlines()
    labels = []
    for index, line in enumerate(lines):
        os.system('clear')
        print(file_path)
        print("\n".join(lines[index:]))

        label = None
        while label is None:
            char_in = readchar()

            if ord(char_in) == 27:
                exit(1)

            if char_in == "a":
                label = "author"
            elif char_in == "c":
                label = "country"
            elif char_in == "e":
                label = "email"
            elif char_in == "o":
                label = "organization"
            elif char_in == "p":
                label = "private"
            elif char_in == "r":
                label = "reviewer"
            elif char_in == "x":
                label = "other"

        labels.append(label)

    assert len(labels) == len(lines), "amount labels does not match lines"

    with open(file_path.replace("abstract_txts", "abstract_labels"), "w") as file:
        file.write("\n".join(labels))