# Basics
import os
import subprocess
import logging
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# SageMaker
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import RealTimePredictor
import boto3

# sklearn
from sklearn.metrics import accuracy_score

# PyTorch
import torch
import torch.utils.data
import torch.optim as optim

# Locals
from project_loadData import read_imdb_data, prepare_imdb_data, preprocess_data, review_to_words
from project_makeDict import build_dict, convert_and_pad_data
from project_trainNN import train
from project_model import LSTMClassifier
from project_test import test_reviews, predict, StringPredictor


# Set a working directory.
working_dir = '/home/ec2-user/SageMaker/sentimentAnalysis/'
# Set a directory where codes are located.
code_dir = '/home/ec2-user/SageMaker/Udacity_ML/deployment/'

# whether or not a new model is trained.
train_new = False
# the name of a trained model to be imported.
trained_job_name = 'sagemaker-pytorch-2020-05-25-02-59-38-123'

# Set a data directory for Imdb raw data and processed data.
data_dir_imdb = os.path.join(working_dir, 'data/imdb/')
data_dir_pytorch = os.path.join(working_dir, 'data/pytorch/')
if not os.path.exists(data_dir_imdb):
    os.makedirs(data_dir_imdb)
if not os.path.exists(data_dir_pytorch):
    os.makedirs(data_dir_pytorch)


# set a logger
log_dir = os.path.join(working_dir, 'log/')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = log_dir + 'project.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s:%(levelname)s:(name)s:%(message)s\n')

file_handler = logging.FileHandler(log_filename)
print('log file {} created'.format(log_filename))

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# Set a SageMaker session.
session = sagemaker.Session()
bucket = session.default_bucket()
prefix = 'sagemaker/sentiment_rnn'
role = sagemaker.get_execution_role()
s3 = boto3.client('s3')


# Load cache file or preprocess data from scratch.
cache_dir = os.path.join(working_dir, 'cache/')
cache_file = 'preprocessed_data.pkl'
with open(cache_dir + cache_file, "rb") as f:
    cache_data = pickle.load(f)
    # Unpack data loaded from cache file
    train_X, test_X, train_y, test_y = (
        cache_data['words_train'],
        cache_data['words_test'],
        cache_data['labels_train'],
        cache_data['labels_test']
    )
    logger.info("Read preprocessed data from a cache file:" + cache_file)


# Build a word dictionary.
cache_file = 'word_dict.pkl'
with open(cache_dir + cache_file, "rb") as f:
    word_dict = pickle.load(f)
    logger.info("Read word_dict from a cache file: " + cache_file)

# Convert lists of words into lists of indices.
train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
test_X, test_X_len = convert_and_pad_data(word_dict, test_X)
logger.info(
    "Training and test data converted and padded.\n\
    For example, train_X[10]: {}\n\
    train_X_len[10]: {}".format(train_X[10], train_X_len[10])
)

# Check if the data are already uploaded to S3.
input_data = 's3://{}/{}'.format(bucket, prefix)
logger.info("input_data: {}". format(input_data))



# Below for PyTorch implementation.


# Train a full model.
logger.info("Train a full PyTorch model.")
estimator = PyTorch(
    entry_point='project_trainNN.py',
    source_dir=code_dir,
    role=role,
    framework_version='0.4.0',
    train_instance_count=1,
    train_instance_type='ml.p2.xlarge',
    hyperparameters={
        'epochs': 10,
        'hidden_dim': 200
    }
)

if train_new is True:
    logger.info("Train from scratch.")
    estimator.fit({'training': input_data})
else:
    logger.info("Load a pre-trained model.")
    estimator = estimator.attach(trained_job_name)


# Deploy the model for a webapp.
estimator2 = PyTorchModel(
    model_data=estimator.model_data,
    role=role,
    framework_version='0.4.0',
    entry_point='predict.py',
    source_dir=os.path.join(code_dir, 'serve'),  # use 'serve/predict.py'
    predictor_cls=StringPredictor
)
logger.info("A model for a webapp is created.")

estimator2_endpoint = estimator2.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)
logger.info("An endpoint for a webapp is created.")

logger.info("Test the first 10 reviews.")
ground, results = test_reviews(
    predictor=estimator2_endpoint,
    data_dir=os.path.join(data_dir_imdb, 'aclImdb'),  # a directory of test data
    stop=10
)
logger.info("Accuracy score: {}".format(accuracy_score(ground, results)))
logger.info()

# Delete the endpoint.
estimator2_endpoint.delete_endpoint()
logger.info("Endpoint deleted.")
