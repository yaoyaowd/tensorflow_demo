import numpy as np
import random
from search.vocabulary_processor import VocabularyProcessor
import sys

reload(sys)
sys.setdefaultencoding("utf-8")


class InputHelper(object):
    def load_tsv_data(self, filepath):
        print "loading training data from:", filepath
        x1 = []
        x2 = []
        y = []
        count = 0
        for line in open(filepath):
            count += 1
            items = line.strip().split("\t")
            if len(items) != 3:
                continue
            x1.append(items[0])
            x2.append(items[2])
            y.append(1)
            if count % 1000000 == 0:
                print "loading %d lines" % count

        neg = [x for x in x2]
        random.shuffle(neg)
        for i in xrange(len(neg)):
            x1.append(x1[i])
            x2.append(neg[i])
            y.append(0)
        return np.asarray(x1), np.asarray(x2), np.asarray(y)

    def batch_iter(self, data, batch_size, num_epochs):
        data = np.asarray(data)
        print data
        print data.shape

        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            shuffled_data = data[np.random.permutation(np.arange(len(data)))]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min(start_index + batch_size, len(data))
                #yield data[start_index:end_index]
                yield shuffled_data[start_index:end_index]

    def load_data(self, filepath, max_document_length, vocab=None):
        x1, x2, y = self.load_tsv_data(filepath)
        vocab_processor = VocabularyProcessor(max_document_length)
        if vocab:
            vocab_processor = vocab
        else:
            print "Building vocabulary"
            vocab_processor.fit(np.concatenate((x1,x2), axis=0))
        x1_num = np.asarray(list(vocab_processor.transform(x1)))
        x2_num = np.asarray(list(vocab_processor.transform(x2)))
        train_set = (x1_num, x2_num, y)
        return train_set, vocab_processor
