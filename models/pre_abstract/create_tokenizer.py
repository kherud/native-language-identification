import os
import re
import json
import tqdm
import glob
from tokenizers import ByteLevelBPETokenizer

abstract_re = re.compile("\s*".join("ABSTRACT") + "|" + "\s*".join("Abstract"))

papers = {}
for file_path in glob.glob("../../data/conferences/*.json"):
    with open(file_path, "r") as file_path:
        conference = json.load(file_path)
        for paper in conference:
            papers[paper["id"]] = paper

for file_path in tqdm.tqdm(glob.glob("../../data/txts/*.txt") + glob.glob("../../data/txts2/*.txt")):
    paper_name = file_path.split("/")[-1].replace(".pdf.txt", "")
    if os.path.exists(f"../../data/pre_abstract_txts/{paper_name}.txt"):
        continue

    with open(file_path) as file:
        text = file.read()

    abstract = None
    mention_count = len(abstract_re.findall(papers[paper_name]["title"]))
    if mention_count > 0:
        mentions = list(abstract_re.finditer(text))
        if len(mentions) >= mention_count:
            abstract = mentions[mention_count]
    else:
        abstract = abstract_re.search(text)

    if abstract is None:
        continue

    with open(f"../../data/pre_abstract_txts/{paper_name}.txt", "w") as file:
        file.write(text[:abstract.start()])

files = glob.glob("../../data/pre_abstract_txts/*.txt")
tokenizer = ByteLevelBPETokenizer(lowercase=True)
tokenizer.train(files, vocab_size=2500, special_tokens=["[PAD]"])
tokenizer.save("tokenizer")