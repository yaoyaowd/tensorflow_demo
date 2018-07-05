import collections
import tensorflow as tf

TRAIN_FILE = '/Users/dwang/Downloads/ptb.train.txt'
TEST_FILE = '/Users/dwang/Downloads/ptb.test.txt'
VALID_FILE = '/Users/dwang/Downloads/ptb.valid.txt'


def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x:(-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data]


def ptb_raw_data():
    word_to_id = _build_vocab(TRAIN_FILE)
    train_data = _file_to_word_ids(TRAIN_FILE, word_to_id)
    valid_data = _file_to_word_ids(VALID_FILE, word_to_id)
    test_data = _file_to_word_ids(TEST_FILE, word_to_id)
    vocabulary_size = len(word_to_id)
    print 'vocabulary size:', vocabulary_size
    return train_data, valid_data, test_data, vocabulary_size


if __name__ == '__main__':
    train, valid, test, vsize = ptb_raw_data()
    print vsize
    print train[:10]
