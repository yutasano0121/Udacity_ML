from tqdm import tqdm
import numpy as np

def build_dict(data, vocab_size=5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.
    word_count = {}  # A dict storing the words that appear in the reviews along with how often they occur
    for words in tqdm(data):  # 'data' is a list of lists.
        for w in words:
            try:
                word_count[w] += 1
            except KeyError:
                word_count[w] = 1

    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.
    sorted_words = [
        w for w in sorted(
            word_count,
            key=word_count.get,
            reverse=True
        )
    ]

    word_dict = {}  # This is what we are building, a dictionary that translates words into integers
    # The -2 is so that we save room for the 'no word'
    for idx, word in enumerate(sorted_words[:vocab_size - 2]):
        # 'infrequent' labels
        word_dict[word] = idx + 2

    return word_dict


# Convert words in sentences into indices in the dictionary
def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 0
    INFREQ = 1  # words not in the dictionary

    working_sentence = [NOWORD] * pad  # a list of 0 with length of pad

    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ

    return working_sentence, min(len(sentence), pad)


# Return np.arrays of converted sentences and their length.
def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []

    # Apply convert_and_pad to each sentence in the data.
    for sentence in tqdm(data):
        converted, length = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(length)

    return np.array(result), np.array(lengths)
