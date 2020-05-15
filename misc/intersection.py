import re
import glob
import tqdm
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import multiprocessing

"""
intersection: {',', ')', '.', '('}
"""


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
alpha_re = re.compile(r'[^a-zA-Z\s]')
space_re = re.compile(r'[\s]+')


def process_file(file_path):
    with open(file_path, "r") as file:
        text = file.read().lower()
        text = space_re.sub(" ", text)
    return set(lemmatizer.lemmatize(token)  # lemmatize (normalize) word
               for token in word_tokenize(text)
               if token not in stop_words)


if __name__ == "__main__":
    file_paths = glob.glob("../data/txts/*.txt")
    with multiprocessing.Pool() as pool:
        documents = list(tqdm.tqdm(pool.imap(process_file, file_paths), total=len(file_paths)))

    vocab = set.union(*documents)

    print(len(documents))
    tokens = {}
    for token in vocab:
        tokens[token] = 0
        for document in documents:
            if token in document:
                tokens[token] += 1

    with open("../tf-idf/token-document-count.txt", "w") as file:
        file.write(f"{len(documents)} documents")
        for token in sorted(tokens, key=tokens.get):
            file.write(f"{token}: {tokens[token]}\n")

    # print(set.intersection(*documents))
