import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names


with open('examples/AAAI12-4.pdf.txt', 'r', encoding="utf8") as f:
    for line in f:
        sentences = nltk.sent_tokenize(line)
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
        chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

        entities = []
        for tree in chunked_sentences:
            entities.extend(extract_entity_names(tree))

        print(entities)
