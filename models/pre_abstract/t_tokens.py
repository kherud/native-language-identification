import glob
from collections import defaultdict

chars = defaultdict(int)
file_paths = glob.glob("../../data/pre_abstract_txts/*.txt")
for file_path in file_paths:
    with open(file_path, "r") as file:
        for c in file.read():
            chars[c] += 1

for char in sorted(chars, key=chars.get):
    print(char, chars[char])
print(len(chars))
print(list(sorted(chars, key=chars.get)))