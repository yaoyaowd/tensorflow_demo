import collections
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data]


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory, converts
    strings to integer ids and performs mini-batching
    of the inputs."""
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary_size = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary_size


def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data, generates batch_size
    pointers into the raw PTB data and allows minibatch
    iteration along these pointers."""
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i : batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("Epoch size = 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps : (i + 1) * num_steps]
        y = data[:, i * num_steps + 1 : (i + 1) * num_steps + 1]
        yield (x, y)
