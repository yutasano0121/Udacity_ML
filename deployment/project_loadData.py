import os
import glob
from tqdm import tqdm

from sklearn.utils import shuffle

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
import re
from bs4 import BeautifulSoup

import pickle


def read_imdb_Data(data_dir):
    data = {}
    labels = {}

    for data_type in tqdm(['train', 'test']):
        data[data_type] = {}  # dependent variables
        labels[data_type] = {}  # tagret variables

        for sentiment in tqdm(['pos', 'neg']):
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []

            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)

            for f in tqdm(files):
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    labels[data_type][sentiment].append(
                        1 if sentiment == 'pos' else 0
                    )  # Make pos and neg into 1 and 0.

            assert len(data[data_type][sentiment]) == len(labels[data][sentiment]), \
                "{} / {} data size does not match label size".format(
                    data_type, sentiment)

    return(data, labels)  # Return a dictionary containing data and labels.


def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""
    # Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']

    # Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)

    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test


train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
logger.info(
    "Data shuffled. \n\
    sIMDb reviews (combined): train = {}, test = {}".format(
        len(train_X),
        len(test_X)
    )
)


def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()

    text = BeautifulSoup(review, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # Convert to lower case
    words = text.split()  # Split string into words
    words = [w for w in words if w not in stopwords.words(
        "english")]  # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words]  # stem

    return words


def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir, cache_file):
    """Convert each review to words; read from cache if available."""

    # Preprocess training and test data to obtain words for each review
    #words_train = list(map(review_to_words, data_train))
    #words_test = list(map(review_to_words, data_test))
    words_train = [review_to_words(review) for review in data_train]
    words_test = [review_to_words(review) for review in data_test]

    # Write to cache file for future runs
    if cache_file is not None:
        cache_data = dict(words_train=words_train, words_test=words_test,
                          labels_train=labels_train, labels_test=labels_test)
        with open(os.path.join(cache_dir, cache_file), "wb") as f:
            pickle.dump(cache_data, f)
        print("Wrote preprocessed data to cache file:", cache_file)

    return words_train, words_test, labels_train, labels_test
