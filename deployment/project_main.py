import os
import subprocess
import logging
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import sagemaker
import torch
import torch.utils.data

from project_loadData import read_imdb_data, prepare_imdb_data, preprocess_data
from project_makeDict import build_dict, convert_and_pad_data


# set a working directory
working_dir = '/home/ec2-user/SageMaker/sagemaker-deployment/'


# set a data directory
data_dir = working_dir + 'data/pytorch/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# set a logger
log_dir = working_dir + 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = log_dir + 'project.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:(name)s:%(message)s\n')

file_handler = logging.FileHandler(log_filename)
print('log file {} created'.format(log_filename))

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# Set a SageMaker session.
session = sagemaker.Session()
bucket = session.default_bucket()
prefix = 'sagemaker/sentiment_rnn'
role=sagemaker.get_execution_role()


# Load cache file or preprocess data from scratch.
cache_dir = working_dir + 'cache/sentiment_analysis/'
cache_file = 'preprocessed_data.pkl'
if os.path.exists(cache_dir):
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
else:
    # Download original data if needed
    if not os.path.exists(working_dir + '/data/aclImdb'):
        subprocess.checl_call(
            "wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            shell=True
        )
        subprocess.check_call(
            "tar -zxf ../data/aclImdb_v1.tar.gz -C ../data",
            shell=True
        )

    # Load raw data
    data, labels = read_imdb_data(working_dir + 'data/aclImdb')
    logger.info(
        "Raw data loaded. \n\
        IMDB reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
            len(data['train']['pos']),
            len(data['train']['neg']),
            len(data['test']['pos']),
            len(data['test']['neg'])
        )
    )

    # Shuffle the data.
    train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
    logger.info(
        "Data shuffled. \n\
        sIMDb reviews (combined): train = {}, test = {}".format(
            len(train_X),
            len(test_X)
        )
    )

    # Preprocess the data and pickle it.
    train_X, test_X, train_y, test_y = preprocess_data(
        train_X, test_X, train_y, test_y,
        cache_dir, cache_file
    )
    logger.info('Train and test data preprocessed and pickled in {}'.format(cache_file))


# Build a word dictionary.
cache_file = 'word_dict.pkl'
if os.path.exists(cache_dir + cache_file):
    with open(cache_dir + cache_file, "rb") as f:
        word_dict = pickle.load(f)
        logger.info("Read word_dict from a cache file: " + cache_file)
else:
    word_dict = build_dict(train_X)
    logger.info("word_dict constructed.")

    with open(cache_dir + cache_file, "wb") as f:
        pickle.dump(word_dict, f)
    logger.info("word_dict pickled: " + cache_file)


# Convert lists of words into lists of indices.
train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
test_X, test_X_len = convert_and_pad_data(word_dict, test_X)
logger.info(
    "Training and test data converted and padded.\n\
    For example, train_X[10]: {}\n\
    train_X_len[10]: {}".format(train_X[10], train_X_len[10])
)


# Save the processed data locally.
pd.concat(  # training data
    [
        pd.DataFrame(train_y),  # labels
        pd.DataFrame(train_X_len),  # lengths of sentences
        pd.DataFrame(train_X)  # data
    ], axis=1
).to_csv(data_dir + 'train.csv', header=False, index=False)

pd.concat(  # test data
    [
        pd.DataFrame(train_X_len),  # lengths of sentences
        pd.DataFrame(train_X)  # data
    ], axis=1
).to_csv(data_dir + 'test.csv', header=False, index=False)

# test labels
pd.DataFrame(test_y).to_csv(data_dir + 'test.csv', header=False, index=False)


# Upload the data to S3.
input_data = session.upload_data(
    path=data_dir,
    bucket=bucket,
    key_prefix=prefix
)


"""
Below for PyTorch implementation.
"""


# Print the neural network to be impremented.
subprocess.check_call(
    'pygmentize {}Project/train/model.py'.format(working_dir),
    shell=True
)

# Load the first 250 rows.
train_sample = pd.read_csv(
    data_dir + 'train.csv',
    header=None, names=None, nrows=250
)

# Turns the dataframe into tensors.
train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze()
train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long()

# Build a dataset.
train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
# Build a dataloader.
train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)
