import os
import logging
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from project_loadData import read_imdb_Data, prepare_imdb_data, review_to_words, preprocess_data
from project_makeDict import build_dict

# set working directory
working_dir = '/home/ec2-user/SageMaker/sagemaker-deployment/'

# set a logger
log_dir = working_dir + 'log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = log_dir + 'project.log'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:(name)s:%(message)s')

file_handler = logging.FileHandler(log_filename)
print('log file {} created'.format(log_filename))

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


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
    data, labels = read_imdb_Data(working_dir + 'data/aclImdb')
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
