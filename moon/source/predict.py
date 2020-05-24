import argparse
import argparse
import os
import sys
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from io import StringIO
from six import BytesIO

# import model
from model import SimpleNet


# Accept and return numpy data.
contentType = 'application/x-npy'


def model_fn(model_dir):
    print("Loading a model.")

    # Load model information.
    model_info = {}  # Is this line needed?
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("Model information: {}".format(model_info))

    # Define a device and a model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNet(
        model_info['input_dim'],
        model_info['hidden_dim'],
        model_info['output_dim']
    )

    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()  # evaluation mode

    print("Model loaded.")

    return model


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')

    if content_type == contentType:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)

    else:
        return Exception('Requested unsupported content type: ' + content_type)


def output_fn(prediction_output, accept):
    print('Serializing the output.')

    if accept == contentType:
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), accept
    else:
        return Exception('Requested unsupported content type: ' + accept)


def predict_fn(input_data, model):
    print("Prediction being made...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.from_numpy(input_data.astype('float32'))
    data = data.to(device)

    model.eval()

    # Apply the model to the input data.
    out = model(data)
    # Convert the result into a numpy array of binary class labels.
    result = out.cpu().detach().numpy

    return result
