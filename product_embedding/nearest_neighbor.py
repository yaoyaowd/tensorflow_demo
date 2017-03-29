import argparse
import numpy as np
import random
import sys
import ujson

from pyflann import *


def load_input(filename, cids, percent):
    cid2values = dict()
    with open(filename) as f:
        cnt = 0
        for l in f:
            items = l.strip().split('\t')
            cid = items[0]
            value = [float(x) for x in items[1].split(',')]
            if cid in cids or random.randint(0, 100) < percent:
                cid2values[cid] = value
            cnt += 1
            if cnt % 100000 == 0:
                print "processed {} lines".format(cnt)
    return cid2values


def load_rep(filename):
    cids = set()
    with open(filename) as f:
        for l in f:
            items = l.strip().split('\t')
            cids.add(items[1])
    return cids


def create_arg_parser():
    parser = argparse.ArgumentParser(prog="knn")
    parser.add_argument("-r", "--rep",
                        help="a list of representitive")
    parser.add_argument("-i", "--input",
                        help="the input file, list of json input")
    return parser


def main(argv):
    parser = create_arg_parser()
    args = parser.parse_args(argv[1:])

    cids = load_rep(args.rep)
    cid2values = load_input(args.input, cids, 101)

    id2cids = dict()
    values = []
    for cid in cids:
        if cid in cid2values:
            id2cids[len(id2cids)] = cid
            values.append(cid2values[cid])

    dataset = np.array(values)
    flann = FLANN()
    params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.95, log_level="info")

    line = raw_input('cid:')
    while line != 'exit':
        cid = line.strip()
        if cid not in cid2values:
            "Cid does not have an embedding"
            line = raw_input('cid:')
            continue
        result, dists = flann.nn_index(np.array(cid2values[cid]), 50, checks=params['checks'])
        print ','.join([id2cids[x] for x in result[0]])
        line = raw_input('cid:')


if __name__ == "__main__":
    main(sys.argv)
