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
from project_model import LSTMClassifier, predict, StringPredictor
from project_test import test_reviews

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
        pd.DataFrame(test_X_len),  # lengths of sentences
        pd.DataFrame(test_X)  # data
    ], axis=1
)
test_X.to_csv(data_dir + 'test.csv', header=False, index=False)

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


# Test NN
device = torch.devie('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(32, 100, 5000).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()

logger.info("Test a PyTorch model.")
train(model, train_sample_dl, 5, optimizer, loss_fn, device)


# Train a full model.
logger.info("Train a full PyTorch model.")
estimator = PyTorch(
    entry_point='project_trainNN.py',
    role=role,
    framework_version='0.4.0',
    train_instance_count=1,
    train_instance_type='ml.p2.xlarge',
    hyperparameters={
        'epochs': 10,
        'hidden_dim': 200
    }
)
estimator.fit({'training': input_data})

# Deploy the model to create an endpoint.
estimator_endpoint = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)
logger.info("Endpoint created.")


# Test the model.
test_X_concat = pd.concat(
    [pd.DataFrame(test_X_len), pd.DataFrame(test_X)],
    axis=1
)

pred_y = predict(
    data=test_X_concat.values,
    deployed_model=estimator_endpoint
)
pred_y = [round(num) for num in pred_y]
logger.info(
    "Trained model tested.\n\
    Accuracy score: {}".format(accuracy_score(test_y, pred_y)
)


# more testing
test_review = "The simplest pleasures in life are the best, and this film is one \
    of them. Combining a rather basic storyline of love and adventure this movie \
    transcends the usual weekend fair with wit and unmitigated charm."
new_words = review_to_words(test_review)
test_data = convert_and_pad_data(word_dict, new_words)
test_result = estimator_endpoint.predict(test_data)
print(test_result)


# Delete the endpoint.
estimator_endpoint.delete_endpoint()
logger.info("Endpoint deleted.")



# Deploy the model for a webapp.
estimator_endpoint2 = PyTorchModel(
    model_data = estimator.model_data,
    role = role,
    framework_version = '0.4.0',
    entry_point = 'predict.py',
    source_dir = 'serve',  # use 'serve/predict.py'
    predictor_cls = StringPredictor
)

predictor = estimator_endpoint2.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)

logger.info("Test the first 100 reviews.")
ground, results = test_reviews(
    predictor=estimator_endpoint2,
    data_dir=data_dir,
    stop=100
)
logger.info("Accuracy score: {}".format(accuracy_score(ground, results)))


# Delete the endpoint.
estimator_endpoint2.delete_endpoint()
logger.info("Endpoint deleted.")
