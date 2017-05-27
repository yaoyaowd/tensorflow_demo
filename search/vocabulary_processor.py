from __future__ import absolute_import
from __future__ import division

import re
import numpy as np
from tensorflow.contrib import learn


TOKENIZER_RE = re.compile(r"\w+")


def tokenizer(iterator):
    for value in iterator:
        yield TOKENIZER_RE.findall(value.lower())


class VocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None,
                 tokenizer_fn=tokenizer):
        self.sup = super(VocabularyProcessor, self)
        self.sup.__init__(max_document_length, min_frequency, vocabulary, tokenizer_fn)

    def transform(self, raw_documents):
        """
        Transform documents to word id matrix
        :param raw_documents:
        :return:
        """
        for document in raw_documents:
            words = TOKENIZER_RE.findall(document.lower())
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(words):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids
