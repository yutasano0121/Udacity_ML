"""
Copied from 'train/model.py' provided by Udacity.
"""

import torch.nn as nn
import numpy as np
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import RealTimePredictor


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x=x.t()
        lengths=x[0, :]
        reviews=x[1:, :]
        embeds=self.embedding(reviews)
        lstm_out, _=self.lstm(embeds)
        out=self.dense(lstm_out)
        out=out[lengths - 1, range(len(lengths))]

        return self.sig(out.squeeze())


class StringPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(StringPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            content_type = 'text/plain'
        )


def predict(data, deployed_model, rows = 512):
    split_array=np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions=np.array([])
    for array in split_array:
        predictions=np.append(predictions, deployed_model.predict(array))
