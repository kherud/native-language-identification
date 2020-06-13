import os
import glob
import random
from readchar import readchar

file_paths = glob.glob("../../data/txts3/*.txt")
file_paths = list(filter(lambda x: not os.path.exists(x.replace("txts3", "references_labels")), file_paths))

random.shuffle(file_paths)

for file_path in file_paths:
    with open(file_path, "r") as file:
        text = file.read()

    document = []
    lines = text.split("\n")
    labels = []
    for index, line in reversed(list(enumerate(lines))):
        os.system('clear')
        print(file_path)
        print("\n".join(lines[:index+1]))

        label = None
        stop = False
        while label is None and not stop:
            char_in = readchar()

            if ord(char_in) == 27:
                exit(1)
            if ord(char_in) == 32:
                stop = True

            if char_in == "a":
                label = "acknowledgment"
            elif char_in == "r":
                label = "references"
            elif char_in == "x":
                label = "other"

        if stop:
            labels = ["other"] * (len(lines) - len(labels)) + labels
            break

        labels.insert(0, label)

    assert len(labels) == len(lines), "amount labels does not match lines"

    with open(file_path.replace("txts3", "references_labels"), "w") as file:
        file.write("\n".join(labels))
    with open(file_path.replace("txts3", "references_txts"), "w") as file:
        file.write("\n".join(lines))