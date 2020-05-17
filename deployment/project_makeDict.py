import numpy as np


def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""

    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.
    word_count = {} # A dict storing the words that appear in the reviews along with how often they occur
    for words in tqdm(train_X):  # train_X is a list of lists.
        for w in tqdm(words):
            try:
                word_count[w] += 1
            except KeyError:
                word_count[w] == 1

    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.
    sorted_words = [w for w in sorted(word_count, key=word_count.get, reverse=True)]

    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels

    return word_dict


word_dict = build_dict(train_X)
logger.info("word_dict constructed.")

with open(cache_dir + 'word_dict.pkl'), "wb") as f:
    pickle.dump(word_dict, f)
logger.info("word_dict pickled: word_dict.pkl")
