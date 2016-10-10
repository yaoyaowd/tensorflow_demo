import re


# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])\t")
_DIGIT_RE = re.compile(br"\d")

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def initialize_vocabulary(data_file, normalize_digits=True, max_vocabulary_size=100000):
    """Initialize vocabulary from file.
    We assume the vocabulary is based on most frequent words.
    """
    with open(data_file) as f:
        vocab = dict()
        for line in f:
            tokens = basic_tokenizer(line)
        for w in tokens:
            word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        word_to_id = dict(zip(vocab_list, range(len(vocab_list))))
    return word_to_id


def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.
  """
  words = basic_tokenizer(sentence)
  if not normalize_digits:
      return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def initialize_a_to_z():
    word_to_id = dict()
    word_to_id[_PAD] = PAD_ID
    word_to_id[_GO] = GO_ID
    word_to_id[_EOS] = EOS_ID
    word_to_id[_UNK] = UNK_ID
    id = UNK_ID
    for x in "abcdefghijklmnopqustuvwxyz ":
        id += 1
        word_to_id[x] = id
    return word_to_id


def string_to_token_ids(str, vocabulary):
    return [vocabulary.get(c, UNK_ID) for c in str]
