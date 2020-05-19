"""
Copied from 'train/model.py' provided by Udacity.
"""

import torch.nn as nn
import numpy as np


class LSTMClassifier(nn.Module):
    def __init__(self, embeddig_dim, hidden_dim, vocab_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in)features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.t()
        lengths = x[0, :]
        reviews = x[1:, :]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]

        return self.sig(out.squeeze())


def predict(data, deployed_model, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = np.array([])
    for array in split_array:
        predictions = np.append(predictions, deployed_model.predict(array))
