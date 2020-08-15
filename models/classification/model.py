from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_layers: int,
                 hidden_size: int,
                 dropout: float = 0.,
                 norm: Optional[nn.Module] = nn.BatchNorm1d,
                 activation: torch.nn.Module = nn.ReLU):
        super(Linear, self).__init__()

        self.hidden_layers = hidden_layers
        self.dropout = dropout

        self.activation = activation

        self.linear0 = nn.Linear(input_size, hidden_size)
        if norm:
            self.norm0 = norm(hidden_size)
        if self.dropout > 0:
            self.dropout0 = nn.Dropout(self.dropout)
        in_size, out_size = 0, hidden_size
        for i in range(1, hidden_layers):
            in_size = hidden_size // (2 ** (i - 1))
            out_size = hidden_size // (2 ** i)
            setattr(self, f"linear{i}", nn.Linear(in_size, out_size))
            if norm:
                setattr(self, f"norm{i}", norm(out_size))
            if self.dropout > 0:
                setattr(self, f"dropout{i}", nn.Dropout(self.dropout))
        setattr(self, f"linear{self.hidden_layers}", nn.Linear(out_size, output_size))

        # for i in range(hidden_layers):
        #     torch.nn.init.xavier_normal_(getattr(self, f"linear{i}").weight)

    def forward(self, x):
        for i in range(self.hidden_layers):
            x = getattr(self, f"linear{i}")(x)
            if self.norm:
                x = getattr(self, f"norm{i}")(x)
            if self.dropout > 0:
                x = getattr(self, f"dropout{i}")(x)

        return getattr(self, f"linear{self.hidden_layers}")(x)


class Transformer(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 n_classes: int = 12,
                 dropout: float = 0.2):
        super().__init__()
        self.num_classes = n_classes

        self.encoder = encoder
        self.dropout = nn.Dropout(p=dropout)
        # self.linear = nn.Linear(self.encoder.config.hidden_size, self.num_classes)
        self.lstm = nn.LSTM(encoder.config.hidden_size, self.num_classes, batch_first=True)

    def forward(self, input_ids, attention_mask):
        # chunks, dim = input_ids.size()
        _, x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = x.unsqueeze(0)

        lstm_out, _ = self.lstm(x)

        return lstm_out[:, -1, :]


class CNN(nn.Module):
    def __init__(self,
                 n_classes: int,
                 vocab_size: int,
                 embedding_dim: int,
                 lstm_dim: int,
                 dropout: float = 0.):
        super(CNN, self).__init__()
        self.n_classes = n_classes

        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.Ks = [2, 3, 4, 5, 6]
        self.n_filters = 20

        self.p_dropout = dropout
        if self.p_dropout > 0:
            self.dropout = nn.Dropout(p=self.p_dropout)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.LeakyReLU()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim // 2, batch_first=True, bidirectional=True)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.n_filters, (K, embedding_dim)) for K in self.Ks])

        self.token_alignment = nn.Linear(lstm_dim, 1)

        self.classifier = nn.Linear(self.n_filters * len(self.Ks), self.n_classes)

    def forward(self, x):
        # get tensor dimensions
        batch, sentences, words = x.size()

        embeddings = self.embeddings(x)
        embeddings = embeddings.view(batch * sentences, 1, words, self.embedding_dim)

        # lstm_out, _ = self.lstm(embeddings)
        cnn_out = [F.relu(conv(embeddings)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        cnn_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_out]  # [(N, Co), ...]*len(Ks)
        cnn_out = torch.cat(cnn_out, dim=1)
        # cnn_out = cnn_out.view(batch, sentences, self.n_filters * len(self.Ks))

        if self.p_dropout > 0:
            cnn_out = self.dropout(cnn_out)

        # calculate attention weighted state vector
        # lstm_out = lstm_out.reshape(batch * sentences * words, self.lstm_dim)
        # alignment = self.token_alignment(lstm_out)
        # alignment = alignment.view(batch * sentences, words)
        # attention = self.softmax(alignment)
        # lstm_out = lstm_out.reshape(batch * sentences, words, self.lstm_dim)
        # attention_vectors = torch.sum(attention.unsqueeze(2) * lstm_out, dim=1)
        # attention_vectors = attention_vectors.reshape(batch * sentences, self.lstm_dim)
        document_vector = torch.sum(cnn_out, dim=0).unsqueeze(0)

        if self.p_dropout > 0:
            document_vector = self.dropout(document_vector)

        return self.classifier(document_vector)


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