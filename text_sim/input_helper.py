from six.moves import cPickle as pickle
import re
import collections


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " ", string)
    string = re.sub(r"n\'t", " ", string)
    string = re.sub(r"\'re", " ", string)
    string = re.sub(r"\'d", " ", string)
    string = re.sub(r"\'ll", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


vocabulary_size = 50000

def load_data_and_labels(data_file):
    print("loading training data from: {}".format(data_file))
    x1_text, x2_text = [], []
    x1_cid, x2_cid = [], []
    count = 0
    with open(data_file) as input:
        for l in input:
            items = l.strip().split("\t")
            if len(items) != 5:
                continue
            x1_cid.append(items[0])
            x1_text.append(clean_str(items[1]))
            x2_cid.append(items[2])
            x2_text.append(clean_str(items[3]))
            count += 1
            if count % 100000 == 0:
                print("loading {} lines".format(count))
    print("load {0} lines in total".format(count))

    words = ' '.join(x1_text + x2_text).split(' ')
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    x1_data, x2_data = [], []
    x1_label, x2_label = [], []
    unk_count = 0
    for i in range(len(x1_text)):
        valid_x1 = False
        tmp1 = []
        for word in x1_text[i].split(' '):
            if word in dictionary:
                valid_x1 = True
            else:
                unk_count += 1
            tmp1.append(dictionary.get(word, 0))
        valid_x2 = False
        tmp2 = []
        for word in x2_text[i].split(' '):
            if word in dictionary:
                valid_x2 = True
            else:
                unk_count += 1
            tmp2.append(dictionary.get(word, 0))
        if valid_x1 and valid_x2:
            x1_data.append(tmp1)
            x1_label.append(x1_cid[i])
            x2_data.append(tmp2)
            x2_label.append(x2_cid[i])
    print("after transform {0} lines in total".format(len(x1_data)))

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return x1_data, x2_data, x1_label, x2_label, count, dictionary, reverse_dictionary

x1_data, x2_data, x1_label, x2_label, count, dictionary, reverse_dictionary = load_data_and_labels("/home/dong/raw_data.tsv")
print('Most common words (+UNK)', count[:100])
print('Least common words (+UNK)', count[-100:])
for i in range(10):
    print(x1_data[i], x2_data[i])

with open("/home/dong/related_dict.pickle", "w") as f:
    pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
with open("/home/dong/related_cnn.pickle", "w") as f:
    pickle.dump({'x1': x1_data,
                 'x1_label': x1_label,
                 'x2':x2_data,
                 'x2_label': x2_label},
                f, pickle.HIGHEST_PROTOCOL)
