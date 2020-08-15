import os
import re
import tqdm
import string
import pickle
import spacy
import glob
import multiprocessing
from lexicalrichness import LexicalRichness

char_re = re.compile(rf"[^{string.printable}]")
abstract_re = re.compile("\s*".join("ABSTRACT") + "|" + "\s*".join("Abstract"))
ack_re = "\s*".join("Acknowledg") + "e?" + "\s*".join("ment") + "(?:\s*s)?\s*"
ack_re += "|" + "\s*".join("ACKNOWLEDG") + "E?" + "\s*".join("MENT") + "(?:\s*s)?\s*"
ack_re = re.compile(ack_re)
ref_re = "\s*".join("Reference") + "\s*s?"
ref_re += "|" + "\s*".join("REFERENCE") + "\s*S?"
ref_re = re.compile(ref_re)

nlp = spacy.load("en_core_web_md")

with open("function-words.txt", "r") as file:
    function_words = set(file.read().split("\n"))

with open("conferences.pkl", "rb") as file:
    conferences = pickle.load(file)


def detect_start(text, title):
    abstract_mention = None
    abstract_title_mentions = abstract_re.findall(title)
    if len(abstract_title_mentions) > 0:
        abstract_mentions = list(abstract_re.finditer(text))
        if len(abstract_mentions) >= len(abstract_title_mentions):
            abstract_mention = abstract_mentions[len(abstract_title_mentions)]
    else:
        abstract_mention = abstract_re.search(text)

    if abstract_mention is None:
        return 800
    else:
        end = min(abstract_mention.end(), 2500)
        return end


def detect_end(text):
    ack_mention = list(ack_re.finditer(text))
    ack_mention = len(text) if len(ack_mention) == 0 else ack_mention[-1].start()
    ref_mention = list(ref_re.finditer(text))
    ref_mention = len(text) if len(ref_mention) == 0 else ref_mention[-1].start()
    return min(ack_mention, ref_mention)


def process_paper(file_path):
    with open(file_path, "r") as file:
        text = file.read()

    name = file_path.split("/")[-1].split(".")[0]
    start = detect_start(text, conferences[name]["title"])
    end = detect_end(text)

    text = char_re.sub("", text[start:end])
    doc = nlp(text)

    lr = LexicalRichness(text)

    document = {
        "n_chars": len(text),
        "n_tokens": 0,
        "n_sentences": 0,
        "word_lengths": {},
        "sentence_lengths": {},
        "sentence_tokens": {},
        "punctuation": {},
        "function_words": {},
        "tokens": [],
        "tags": [],
        "pos": [],
        "metrics": {}
    }

    for metric in ["cttr", "rttr", "ttr", "Dugast", "Herdan", "Maas", "Summer"]:
        try:
            document["metrics"][metric] = getattr(lr, metric)
        except:
            document["metrics"][metric] = 0
    for metric in ["hdd", "mattr", "msttr", "mtld"]:
        try:
            document["metrics"][metric] = getattr(lr, metric)()
        except:
            document["metrics"][metric] = 0

    for sent in doc.sents:
        for token in sent:
            document["tokens"].append(token.lemma_)
            document["pos"].append(token.pos_)
            document["tags"].append(token.tag_)
            # document["tokens"].setdefault(token.lower(), 0)
            # document["tokens"][token.lower()] += 1
            document["word_lengths"].setdefault(len(token), 0)
            document["word_lengths"][len(token)] += 1
            document["n_tokens"] += 1
            if token.lemma_ in function_words:
                document["function_words"].setdefault(token.lemma_, 0)
                document["function_words"][token.lemma_] += 1
        document["sentence_lengths"].setdefault(len(sent.text), 0)
        document["sentence_lengths"][len(sent.text)] += 1
        document["sentence_tokens"].setdefault(len(sent), 0)
        document["sentence_tokens"][len(sent)] += 1
        document["n_sentences"] += 1

    for char in string.punctuation:
        document["punctuation"][char] = text.count(char)

    with open(f"../data/pkls/{name}.pkl", "wb") as file:
        pickle.dump(document, file)


if __name__ == "__main__":
    file_paths = glob.glob("../data/txts_cleaned/*.txt")
    file_paths = [file_path for file_path in file_paths if
                  not os.path.exists(file_path.replace("txts_cleaned/", "pkls/").replace(".txt", ".pkl"))]
    with multiprocessing.Pool(processes=12) as pool:
        for x in tqdm.tqdm(pool.imap(process_paper, file_paths), total=len(file_paths)):
            pass
