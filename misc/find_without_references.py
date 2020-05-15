import re
import glob
import tqdm

space_re = re.compile(r"[\s]+")

for file_path in tqdm.tqdm(glob.glob("../data/txts/*.txt")):
    with open(file_path, "r") as file:
        text = file.read().lower()

    text = space_re.sub("", text)

    if "reference" not in text:
        print(file_path)