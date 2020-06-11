import os
import re
import glob

all_lines, all_labels = [], []
for file_path in glob.glob("../../data/pre_abstract_labels/*.txt"):
    file_name = file_path.split("/")[-1]
    with open(file_path, "r") as file:
        labels = file.readlines()
    try:
        with open(os.path.join("../../data/pre_abstract_txts/", file_name), "r") as file:
            text = file.readlines()
    except FileNotFoundError:
        continue

    n_lines = min(len(text), len(labels))
    all_lines.extend(text[:n_lines])
    all_labels.extend(labels[:n_lines])

department_re = re.compile("(?:,\s*)?[^,]*Department[^,]*(?:,)")

for line, label in zip(all_lines, all_labels):
    if label.strip() == "organization" and "Department" in line:
        # print(line.strip())
        print(department_re.findall(line.strip()))

