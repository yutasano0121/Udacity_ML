import torch
import torch.nn as nn
import torch.nn.functional as functional


class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        # linear layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Add a hidden layer, with relu activation function.
        x = functional.relu(self.fc1(x))
        # Add a dropout layer.
        x = self.dropout(x)
        # Make an output layer, followed by Sigmoid transformation.
        x = functional.sig(self.fc2(x))

        return x
