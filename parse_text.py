# import nltk
# from nltk.corpus import stopwords

# stop = stopwords.words("english")
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

import glob
import spacy

# nlp = spacy.load("en_trf_robertabase_lg")
# nlp = spacy.load("en_core_web_lg")
nlp = spacy.load("en_core_web_md", disable=["tagger", "parser"])
print(nlp.pipeline)

def parse_text(file_name):
    with open(file_name, "r") as file:
        text = file.read().replace("\n", " ")

    doc = nlp(text)
    labels = set()
    for token in doc:
        print(token, token.like_email)
    for entity in doc.ents:
        print(entity.text, entity.label_)

    print(labels)


if __name__ == "__main__":
    parse_text("data/txts/AAAI12-4.pdf.txt")