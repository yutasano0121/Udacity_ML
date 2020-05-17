import os
import logging
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

working_dir = '/home/ec2-user/SageMaker/sagemaker-deployment/'
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
        words_train, words_test, labels_train, labels_test = (
            cache_data['words_train'],
            cache_data['words_test'],
            cache_data['labels_train'],
            cache_data['labels_test']
        )
        logger.info("Read preprocessed data from a cache file:", cache_file)
else:
    import project_loadData


cache_file = 'word_dict.pkl'
if os.path.exists(cache_dir + cache_file):
    with open(cache_dir + cache_file, "rb") as f:
        word_dict = pickle.load(f)
        logger.info("Read word_dict from a cache file:", cache_file)
else:
    import project_makeDict
