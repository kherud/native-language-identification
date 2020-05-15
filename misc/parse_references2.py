import re
import glob
import tqdm
import string

chars = "A-Za-z\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF\-"
space_re = re.compile(r"\s+")
noise_re = re.compile(f"[^{string.printable}\x7f-\xffÀ-ž\u0370-\u03FF\u0400-\u04FF]")
refer_re = re.compile(r"([(\[](?:\w[^\d()\[\]]+\s[(\[]?[0-9]{4}[A-Za-z]?[(\[]?[;,\s]*)+[)\]])")
et_al_references = re.compile(rf"[{chars}]+\set\.?\s*al\.?\s+(?:[(\[][0-9]{{4}}[)\]])?")
and_references = re.compile(rf"(?:[{chars}]+,?\s+(?:(?:and)|&)\s+)+[{chars}]+[,\s]*(?:[(\[][0-9]{{4}}[A-Za-z]?[)\]]|[0-9]{{4}}[A-Za-z]?)")  # [(\[][0-9]{4}[A-Za-z]?[)\]]
# and_references = re.compile("(?:[" + chars + "]+,?\s+(?:(?:and)|&)\s+)+[" + chars + "]+[,\s]*(?:[(\[][0-9]{4}[A-Za-z]?[)\]]|[0-9]{4}[A-Za-z]?)")  # [(\[][0-9]{4}[A-Za-z]?[)\]]
# test_references = re.compile(r"(\([^)]*,\s*\d\d\d\d(\)|(;[^)])*)\))|([(\[][^)]*,\s*\d\d\d\d([(\[]|(;[^)]*))$)|(^[^()]*,\s*\d\d\d\d(\)|(;[^)])*)\))|(^\s*\d\d\d\d\))")
# et_al_references = re.compile(r"et\.?\s*al\.?\s*")
# test_regex = re.compile(r"[\w.-]+@[\w-]+\.[\w-]+")
for paper in tqdm.tqdm(glob.glob("../data/txts/*.txt")):
    with open(paper, "r") as file:
        text = file.read()

    text = space_re.sub(" ", text)  # replace multiple whitespaces with single space
    text = noise_re.sub("", text)

    all_references = []
    for x in refer_re.finditer(text):
        start, end = x.span()
        all_references.append((text[start:end], start, end))
    for x in et_al_references.finditer(text):
        start, end = x.span()
        all_references.append((text[start:end], start, end))
    for x in and_references.finditer(text):
        start, end = x.span()
        all_references.append((text[start:end], start, end))

    final_references = set()
    for refer_a in all_references:
        for refer_b in all_references:
            if refer_a[1] <= refer_b[1] < refer_a[2] and refer_b[2] <= refer_a[2]:
                start = min(refer_a[1], refer_b[1])
                end = max(refer_a[2], refer_b[2])
                final_references.add(text[start:end])

    print(final_references)
    # print(references)

    result = et_al_references.findall(text)
    # for x in result:
    #     print(result)
    # if len(result) == 0:
    #     print(paper)