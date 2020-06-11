import torch
import torch.nn as nn


class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim, n_classes, dropout):
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.n_classes = n_classes

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.td_lstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True)
        self.alignment = nn.Linear(lstm_dim, 1)
        self.lstm = nn.LSTM(lstm_dim, lstm_dim // 4, batch_first=True, bidirectional=True)
        self.classifier = nn.LSTM(lstm_dim // 2, self.n_classes, batch_first=True)

    def forward(self, x):
        # get tensor dimensions
        batch, sentences, words = x.size()

        embeddings = self.embeddings(x)
        embeddings = embeddings.view(batch * sentences, words, self.embedding_dim)
        # embeddings = self.dropout(embeddings)

        # output shape [batches * sentences, words, lstm_dim]
        td_lstm_out, _ = self.td_lstm(embeddings)

        # calculate attention weighted state vector
        td_lstm_out = td_lstm_out.reshape(batch * sentences * words, self.lstm_dim)
        alignment = self.tanh(self.alignment(td_lstm_out))
        alignment = alignment.view(batch * sentences, words)
        attention = self.softmax(alignment)
        td_lstm_out = td_lstm_out.reshape(batch * sentences, words, self.lstm_dim)
        # [sentences,atttention_per_word] x [sentences,words,lstm_dim] -> [sentences,lstm_dim]
        attention_vectors = torch.einsum("sa,sal->sl", attention, td_lstm_out)
        attention_vectors = attention_vectors.view(batch, sentences, self.lstm_dim)

        # get last hidden state alternatively
        # td_lstm_out = td_lstm_out[:, -1, :]
        # td_lstm_out = td.lstm_out.view(batch, sentences, self.lstm_dim)

        lstm_out, _ = self.lstm(attention_vectors)
        # lstm_out = self.dropout(lstm_out)

        classifier_out, _ = self.classifier(lstm_out)

        return classifier_out

    def annotate(self, classes, tokenizer, text):
        lines = text.split("\n")

        tokenized = [x.ids for x in tokenizer.encode_batch(lines)]

        # padding
        max_tokens = max(len(sentence) for sentence in tokenized)
        for sentence in range(len(tokenized)):
            for _ in range(max_tokens - len(tokenized[sentence])):
                tokenized[sentence].insert(0, 0)

        predictions = self.forward(torch.tensor([tokenized]))
        predictions = torch.argmax(predictions[0], -1)
        predictions = [classes[prediction.item()] for prediction in predictions]

        return zip(lines, predictions)