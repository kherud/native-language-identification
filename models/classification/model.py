import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self,
                 n_classes: int,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.,
                 norm=False):
        super(LSTM, self).__init__()
        self.n_classes = n_classes

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.p_dropout = dropout
        if self.p_dropout > 0:
            self.dropout = nn.Dropout(p=self.p_dropout)

        self.norm = norm
        if self.norm:
            self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
            self.layer_norm = nn.LayerNorm(self.hidden_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.LeakyReLU()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)

        self.token_alignment = nn.Linear(hidden_dim, 1)
        self.line_alignment1 = nn.Linear(hidden_dim, 20)
        self.line_alignment2 = nn.Linear(20, 1)

        self.classifier = nn.Linear(hidden_dim, self.n_classes)

    def forward(self, x):
        # get tensor dimensions
        batch, sentences, words = x.size()

        embeddings = self.embeddings(x)
        embeddings = embeddings.view(batch * sentences, words, self.embedding_dim)

        lstm_out, _ = self.lstm(embeddings)

        # calculate attention weighted state vector
        lstm_out = lstm_out.reshape(batch * sentences * words, self.hidden_dim)
        alignment = self.token_alignment(lstm_out)
        alignment = alignment.view(batch * sentences, words)
        attention = self.softmax(alignment)
        lstm_out = lstm_out.reshape(batch * sentences, words, self.hidden_dim)
        attention_vectors = torch.sum(attention.unsqueeze(2) * lstm_out, dim=1)
        attention_vectors = attention_vectors.reshape(batch * sentences, self.hidden_dim)

        if self.norm:
            attention_vectors = self.batch_norm(attention_vectors)
        if self.p_dropout > 0:
            attention_vectors = self.dropout(attention_vectors)

        # calculate attention weighted state vector
        # alignment = self.relu(self.line_alignment1(attention_vectors))
        # alignment = self.line_alignment2(alignment)
        # alignment = alignment.view(batch, sentences)
        # attention = self.softmax(alignment)
        # attention = attention.view(batch * sentences, 1)
        # attention_vector = torch.sum(attention * attention_vectors, axis=0)
        # attention_vector = attention_vector.view(batch, self.hidden_dim)
        attention_vectors = attention_vectors.view(batch, sentences, self.hidden_dim)
        attention_vector = torch.sum(attention_vectors, dim=1)

        if self.norm:
            attention_vector = self.layer_norm(attention_vector)
        if self.p_dropout > 0:
            attention_vector = self.dropout(attention_vector)

        return self.classifier(attention_vector)