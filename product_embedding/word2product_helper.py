"""
Input words and output products.
"""
import tensorflow as tf


LIMIT = 50000

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "Input filename")
flags.DEFINE_string("output", "", "Output filename")


word_ids = {}
word_freq = {}
id_mapping = {}
p2w = {}    # pid to word id
p2t = {}    # pid to title


def main():
    with open(FLAGS.input) as input:
        for line in input:
            items = line.strip().split('\t')
            if len(items) != 2:
                continue

            id = items[0]
            words = items[1].lower().split(' ')
            p2t[id] = items[1]
            for word in words:
                if word not in word_ids:
                    word_ids[word] = len(word_ids) + 1
                    word_freq[word] = 0
                word_freq[word] = word_freq[word] + 1
            if id not in p2w:
                p2w[id] = [word_ids[word] for word in words]

    id = 0
    for k in sorted(word_freq, key=word_freq.get, reverse=True):
        id += 1
        if id > LIMIT:
            break
        id_mapping[word_ids[k]] = id

    with open(FLAGS.output, 'w') as output:
        for pid in p2w:
            words = p2w[pid]
            new_words = []
            for w in words:
                if w in id_mapping:
                    new_words.append(id_mapping[w])
            if len(new_words) > 0:
                line = pid + '\t' + '\t'.join([str(i) for i in new_words]) + '#' + p2t[pid]
                output.write(line + '\n')


if __name__ == '__main__':
    main()
