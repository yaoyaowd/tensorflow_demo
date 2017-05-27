import re
import tensorflow as tf
from collections import defaultdict

tf.flags.DEFINE_string("input", "", "input file path")
tf.flags.DEFINE_string("output", "", "output file path")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

TOKENIZER_RE = re.compile(r"\w+")


query_tokens = defaultdict(int)
title_tokens = defaultdict(int)
candidates = set()

def clean_tsv_data(input, output):
    count = 0
    for line in open(input):
        count += 1
        items = line.strip().split("\t")
        if len(items) != 3:
            continue
        query = TOKENIZER_RE.findall(items[0].lower())
        title = TOKENIZER_RE.findall(items[2].lower())
        for q in query:
            if len(q) < 20:
                query_tokens[q] = query_tokens[q] + 1
        for t in title:
            if len(t) < 20:
                title_tokens[t] = title_tokens[t] + 1
        if count % 1000000 == 0:
            print "loading %d lines" % count

    for q in query_tokens:
        if query_tokens[q] >= 10:
            candidates.add(q)
    for q in title_tokens:
        if title_tokens[q] >= 10:
            candidates.add(q)
    print 'total candidates: {}'.format(len(candidates))

    count = 0
    with open(output, 'w') as out:
        for line in open(input):
            count += 1
            items = line.strip().split("\t")
            if len(items) != 3:
                continue
            query = TOKENIZER_RE.findall(items[0].lower())
            title = TOKENIZER_RE.findall(items[2].lower())
            new_query = []
            new_title = []
            for q in query:
                if q in candidates:
                    new_query.append(q)
            for t in title:
                if t in candidates:
                    new_title.append(t)
            if len(new_query) > 0 and len(new_title) > 0:
                out.write('\t'.join([' '.join(new_query), items[1], ' '.join(new_title)]) + '\n')
            if count % 1000000 == 0:
                print "loading %d lines" % count


clean_tsv_data(FLAGS.input, FLAGS.output)