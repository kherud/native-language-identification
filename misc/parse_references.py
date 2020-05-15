import re
import glob
import tqdm
import string

space_re = re.compile(r"[\s]+")
noise_re = re.compile(f"[^{string.printable}\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF]")
# references = re.compile(r"[(\[][\w\s\.,\&]+\(?[0-9]{4}[A-Za-z]?\)?[)\]]")
# references = re.compile(r"[(\[][\w\s.,&]+\(?[0-9]{4}[A-Za-z]?(?:[,\s\w]*[0-9.]+)\)?[)\]]")
# references = re.compile(r"[(\[](\w[A-Za-z\s.,&;]+[(\[]?[0-9]{4}[A-Za-z]?[(\[]?)\1*[)\]]")
refer_re = re.compile(r"([(\[](?:\w[^\d()\[\]]+\s[(\[]?[0-9]{4}[A-Za-z]?[(\[]?[;,\s]*)+[)\]])")
# references = re.compile(r"\(|\[([A-Za-z\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF\s.,&]+\(?[0-9]{4}[A-Za-z]?\)?[;,\s]?)+\)|\]")

for paper in tqdm.tqdm(glob.glob("data/txts/*.txt")):
    with open(paper, "r") as file:
        text = file.read()

    text = space_re.sub(" ", text)  # replace multiple whitespaces with single space
    text = noise_re.sub("", text)

    result = refer_re.findall(text)

    print(text)

    for x in result:
        print(x)
    # print(result)
    # if len(result) == 0:
    #     print(paper)