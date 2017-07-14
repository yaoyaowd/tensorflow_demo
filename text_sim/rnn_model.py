from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import time

from six.moves import cPickle as pickle
from random import randint, sample, shuffle
from sklearn.metrics import roc_auc_score


tf.flags.DEFINE_string("input", "", "The input data source")
tf.flags.DEFINE_integer("limit", 1500000, "Total input size")
tf.flags.DEFINE_integer("num_steps", 1000000, "Total training steps")
tf.flags.DEFINE_integer("sequence_length", 50, "Maximal sentence length")
tf.flags.DEFINE_integer("vocab_size", 50001, "Maximal vocabulary size")
tf.flags.DEFINE_integer("pad", 50000, "Pad token id")
tf.flags.DEFINE_integer("valid_size", 1000, "Validation size")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def last_relevant(output, length):
    b_size = tf.shape(output)[0]
    max_len = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    idx = tf.range(0, b_size) * max_len + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, idx)
    return relevant


class TextRNN(object):
    def __init__(self,
                 seq_length=50,
                 vocab_size=50000,
                 embed_size=128,
                 num_layers=1,
                 learning_rate=0.001,
                 use_lstm=True,
                 max_gradient_norm=5.0):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.learning_rate = tf.Variable(float(learning_rate), name='lr', trainable=False)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.use_lstm = use_lstm
        self.max_gradient_norm = max_gradient_norm

        self.input_x1 = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x1')
        self.input_x2 = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x2')
        self.input_y = tf.placeholder(tf.float32, [None], name='input_y')
        self.length_x1 = tf.placeholder(tf.int32, [None], name='length_x1')
        self.length_x2 = tf.placeholder(tf.int32, [None], name='length_x2')
        self.word_embed = tf.Variable(
            tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0), name='w')
        self.x1_embed = tf.nn.embedding_lookup(self.word_embed, self.input_x1, name='x1_embed')
        self.x2_embed = tf.nn.embedding_lookup(self.word_embed, self.input_x1, name='x2_embed')

        # initialize shared encoder
        with tf.variable_scope('shared_encoder'):
            if self.use_lstm:
                single_cell = tf.contrib.rnn.LSTMCell(self.embed_size, forget_bias=1.0, use_peepholes=False)
            else:
                single_cell = tf.contrib.rnn.GRUCell(self.embed_size)
            cell = single_cell
            if self.num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell] * self.num_layers)
            self.M = tf.Variable(
                tf.random_uniform([self.embed_size, self.embed_size], -1.0, 1.0))

            x1_output, _ = tf.nn.dynamic_rnn(cell, self.x1_embed, sequence_length=self.length_x1, dtype=tf.float32)
            x1_last_output = last_relevant(x1_output, self.length_x1)
            self.x1_embedding = tf.matmul(x1_last_output, self.M)

        with tf.variable_scope('shared_encoder', reuse=True):
            x2_output, _ = tf.nn.dynamic_rnn(cell, self.x2_embed, sequence_length=self.length_x2, dtype=tf.float32)
            x2_last_output = last_relevant(x2_output, self.length_x2)
            self.x2_embedding = tf.matmul(x2_last_output, self.M)

        with tf.variable_scope('loss'):
            self.distance = tf.sqrt(tf.reduce_sum(
                tf.square(self.x1_embedding - self.x2_embedding) + 1e-6, 1, keep_dims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.x1_embedding), 1, keep_dims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.x2_embedding), 1, keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
            self.d = self.input_y * self.distance + \
                     (1 - self.input_y) * tf.square(tf.maximum((1 - self.distance), 0))
            self.loss = tf.reduce_mean(self.d)

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_gradient_norm)
        self.train = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)


with open(FLAGS.input, 'rb') as f:
    data = pickle.load(f)
    x1_data = data['x1']
    x2_data = data['x2']
idx = [i for i, _ in enumerate(x1_data)]
idx = sorted(sample(idx, FLAGS.limit))
x1_data = [x1_data[id] for id in idx]
x2_data = [x2_data[id] for id in idx]
y_data = [1] * len(x1_data)
print("Sampled data size: {}".format(len(x1_data)))

num_negative = 2
total_example = len(x1_data)
for id in range(total_example):
    for i in range(num_negative):
        r = randint(0, total_example - 1)
        while abs(id - r) <= 100:
            r = randint(0, total_example - 1)
        x1_data.append(x1_data[id])
        x2_data.append(x2_data[r])
        y_data.append(0)
idx = [i for i, _ in enumerate(x1_data)]
shuffle(idx)
print("Expanded data size: {}".format(len(y_data)))

valid_x1_data = [x1_data[i] for i in idx[:1000]]
valid_x2_data = [x2_data[i] for i in idx[:1000]]
valid_y_data = [y_data[i] for i in idx[:1000]]
idx = idx[1000:]

seq_length = FLAGS.sequence_length
with tf.Graph().as_default(), tf.Session().as_default() as sess:
    rnn = TextRNN(seq_length=seq_length,
                  vocab_size=FLAGS.vocab_size,
                  embed_size=128)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sess.run(tf.global_variables_initializer())

    prev_loss = []
    total_loss = 0
    for i in range(FLAGS.num_steps):
        train_ids = sample(idx, 64)
        input_x1 = np.asarray(
            [[v for v in x1_data[id][:seq_length] + [FLAGS.pad] * (seq_length - len(x1_data[id][:seq_length]))]
             for id in train_ids])
        input_x2 = np.asarray(
            [[v for v in x2_data[id][:seq_length] + [FLAGS.pad] * (seq_length - len(x2_data[id][:seq_length]))]
             for id in train_ids])
        length_x1 = np.asarray([min(seq_length, len(x1_data[id])) for id in train_ids])
        length_x2 = np.asarray([min(seq_length, len(x2_data[id])) for id in train_ids])
        input_y = np.asarray([float(y_data[id]) for id in train_ids])
        _, step, loss = sess.run(
            [rnn.train, rnn.global_step, rnn.loss],
            {
                rnn.input_x1: input_x1,
                rnn.input_x2: input_x2,
                rnn.length_x1: length_x1,
                rnn.length_x2: length_x2,
                rnn.input_y: input_y
            }
        )
        total_loss += loss

        if (i + 1) % 100 == 0:
            path = rnn.saver.save(sess, checkpoint_prefix, global_step=step)
            print("step {0} loss {1} model {2}".format(step, total_loss, path))
            prev_loss.append(total_loss)
            total_loss = 0

            input_x1 = np.asarray(
                [[v for v in x1[:seq_length] + [FLAGS.pad] * (seq_length - len(x1[:seq_length]))]
                 for x1 in valid_x1_data])
            input_x2 = np.asarray(
                [[v for v in x2[:seq_length] + [FLAGS.pad] * (seq_length - len(x2[:seq_length]))]
                 for x2 in valid_x2_data])
            length_x1 = np.asarray([min(seq_length, len(x1)) for x1 in valid_x1_data])
            length_x2 = np.asarray([min(seq_length, len(x2)) for x2 in valid_x2_data])
            input_y = np.asarray([float(v) for v in valid_y_data])
            reversed_y = np.asarray([1.0 - float(v) for v in valid_y_data])
            loss, dist = sess.run(
                [rnn.loss, rnn.distance],
                {
                    rnn.input_x1: input_x1,
                    rnn.input_x2: input_x2,
                    rnn.length_x1: length_x1,
                    rnn.length_x2: length_x2,
                    rnn.input_y: input_y
                }
            )
            auc = roc_auc_score(y_true=reversed_y, y_score=dist)
            print("VALIDATION: step {0} loss {1} auc {2}".format(step, loss, auc))
