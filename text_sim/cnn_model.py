import numpy as np
import tensorflow as tf
import math
import os
import time

from six.moves import cPickle as pickle
from random import sample


tf.flags.DEFINE_string("input", "", "The input data source")
tf.flags.DEFINE_integer("limit", 3500000, "Total input size")
tf.flags.DEFINE_integer("num_steps", 500000, "Total training steps")
tf.flags.DEFINE_integer("sequence_length", 50, "Maximal sentence length")
tf.flags.DEFINE_integer("vocab_size", 50001, "Maximal vocabulary size")
tf.flags.DEFINE_integer("pad", 50000, "Pad token id")
tf.flags.DEFINE_integer("valid_size", 1000, "Validation size")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


class TextCNN(object):
    def __init__(self,
                 sequence_length,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters):
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name='input_x1')
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name='input_x2')
        # self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.input_y = tf.placeholder(tf.int32, [None], name="input_y")

        # embedding lookup
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='w')
            self.embedded_x1 = tf.nn.embedding_lookup(W, self.input_x1)
            self.embedded_x2 = tf.nn.embedding_lookup(W, self.input_x2)
            self.embedded_x1_expanded = tf.expand_dims(self.embedded_x1, -1)
            self.embedded_x2_expanded = tf.expand_dims(self.embedded_x2, -1)

        # create a convolution + max pool layer
        pooled_outputs_x1 = []
        pooled_outputs_x2 = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-{0}'.format(filter_size)):
                filter_shape = [filter_size, embedding_size, 1, num_filters[i]]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters[i]]), name='b')
                conv_x1 = tf.nn.conv2d(
                    self.embedded_x1_expanded,
                    W,
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='conv_x1')
                h1_x1 = tf.nn.relu(tf.nn.bias_add(conv_x1, b), name='relu_x1')
                pooled_x1 = tf.nn.max_pool(
                    h1_x1,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='pool_x1')
                pooled_outputs_x1.append(pooled_x1)
                conv_x2 = tf.nn.conv2d(
                    self.embedded_x2_expanded,
                    W,
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='conv_x2')
                h1_x2 = tf.nn.relu(tf.nn.bias_add(conv_x2, b), name='relu_x2')
                pooled_x2 = tf.nn.max_pool(
                    h1_x2,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='pool_x2')
                pooled_outputs_x2.append(pooled_x2)

        feature_size = sum(num_filters)
        self.x1_pool = tf.concat(pooled_outputs_x1, 3)
        self.x1_pool_flat = tf.reshape(self.x1_pool, [-1, feature_size])
        self.x2_pool = tf.concat(pooled_outputs_x2, 3)
        self.x2_pool_flat = tf.reshape(self.x2_pool, [-1, feature_size])

        # transform to lower dimension
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([feature_size, embedding_size], stddev=0.1), name='W')
            b = tf.Variable(tf.zeros(embedding_size))
            self.final_x1 = tf.nn.xw_plus_b(self.x1_pool_flat, W, b, name='x1_embed')
            self.final_x2 = tf.nn.xw_plus_b(self.x2_pool_flat, W, b, name='x2_embed')

        # optimize normalized cossin similarity
        normal_x1 = tf.nn.l2_normalize(self.final_x1, dim=1)
        normal_x2 = tf.nn.l2_normalize(self.final_x2, dim=1)
        self.similarity = tf.matmul(normal_x1, normal_x2, transpose_b=True)
        self.loss =tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.similarity, labels=self.input_y))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdagradOptimizer(0.01).minimize(self.loss, global_step=self.global_step)

"""
# Optimize distance

        self.distance = tf.sqrt(tf.reduce_sum(tf.square(self.final_x1 - self.final_x2) + 1e-6, 1, keep_dims=True))
        self.distance = tf.div(self.distance,
                               tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.final_x1), 1, keep_dims=True)),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.final_x2), 1, keep_dims=True))))
        self.distance = tf.reshape(self.distance, [-1], name="distance")
        self.d = self.input_y * self.distance + \
                 (1 - self.input_y) * tf.square(tf.maximum((1 - self.distance), 0))
        self.loss = tf.reduce_mean(self.d)

        normal_x1 = tf.nn.l2_normalize(self.final_x1, dim=1)
        normal_x2 = tf.nn.l2_normalize(self.final_x2, dim=1)
        self.similarity = tf.matmul(normal_x1, normal_x2, transpose_b=True)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdagradOptimizer(0.001).minimize(self.loss, global_step=self.global_step)
"""


with open(FLAGS.input, 'rb') as f:
    data = pickle.load(f)
    x1_data = data['x1']
    x2_data = data['x2']
    x1_label = data['x1_label']
    x2_label = data['x2_label']
idx = [i for i, _ in enumerate(x1_data)]
idx = sample(idx, FLAGS.limit)
x1_data = [x1_data[id] for id in idx]
x2_data = [x2_data[id] for id in idx]
x1_label = [x1_label[id] for id in idx]
x2_label = [x2_label[id] for id in idx]
print("Sampled data size: {}".format(len(x1_data)))

valid_x1_data = x1_data[:2048]
valid_x2_data = x2_data[:2048]
valid_x1_label = x1_label[:2048]
valid_x2_label = x2_label[:2048]
valid_idx = [i for i in range(2048)]

x1_data = x1_data[2048:]
x2_data = x2_data[2048:]
x1_label = x1_label[2048:]
x2_label = x2_label[2048:]
y_data = [1] * len(x1_data)
idx = [i for i, _ in enumerate(x1_data)]

"""
num_negative = 2
total_example = len(x1_data)
for id in range(total_example):
    for i in range(num_negative):
        r = randint(0, total_example - 1)
        x1_data.append(x1_data[id])
        x2_data.append(x2_data[r])
        y_data.append(0)
"""

print("Expanded data size: {}".format(len(y_data)))

seq_length = FLAGS.sequence_length
with tf.Graph().as_default(), tf.Session().as_default() as sess:
    cnn = TextCNN(
        sequence_length=FLAGS.sequence_length,
        vocab_size=FLAGS.vocab_size,
        embedding_size=128,
        filter_sizes=[2, 3, 4, 5],
        num_filters=[256, 128, 64, 64])

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summary
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    train_summary_op = tf.summary.merge([loss_summary])
    train_summer_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summer_dir, sess.graph)

    # Models
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    total_loss = 0
    sess.run(tf.global_variables_initializer())
    for i in range(FLAGS.num_steps):
        offset = i * 128 % len(x1_data)
        limit = i * 128 % len(x1_data) + 128
        limit = len(x1_data) if limit >= len(x1_data) else limit
        input_x1 = np.asarray(
            [[v for v in x1_data[id][:seq_length] + [FLAGS.pad] * (seq_length - len(x1_data[id][:seq_length]))]
             for id in range(offset, limit)])
        input_x2 = np.asarray(
            [[v for v in x2_data[id][:seq_length] + [FLAGS.pad] * (seq_length - len(x2_data[id][:seq_length]))]
             for id in range(offset, limit)])
        # y = np.asarray([float(y_data[id]) for id in train_ids])
        y = np.asarray([id - offset for id in range(offset, limit)])
        _, step, summaries, loss = sess.run(
            [cnn.optimizer, cnn.global_step, train_summary_op, cnn.loss],
            {cnn.input_x1: input_x1,
             cnn.input_x2: input_x2,
             cnn.input_y: y})
        train_summary_writer.add_summary(summaries, step)

        total_loss += loss
        if (i + 1) % 100 == 0:
            print("step {0} loss {1}".format(step, total_loss))
            path = saver.save(sess, checkpoint_prefix, global_step=step)
            total_loss = 0
            if math.isnan(loss):
                break

            input_x1 = np.asarray(
                [[v for v in x1[:seq_length] + [FLAGS.pad] * (seq_length - len(x1[:seq_length]))]
                 for x1 in valid_x1_data])
            input_x2 = np.asarray(
                [[v for v in x2[:seq_length] + [FLAGS.pad] * (seq_length - len(x2[:seq_length]))]
                 for x2 in valid_x2_data])
            similarity = sess.run(
                [cnn.similarity],
                {cnn.input_x1: input_x1,
                 cnn.input_x2: input_x2})[0]
            pred = np.flip(np.argsort(similarity, axis=1), axis=1)

            acc1, acc5, acc10, tot = 0.0, 0.0, 0.0, 0.0
            for t in range(pred.shape[0]):
                tot += 1.0
                for rk in range(pred.shape[1]):
                    v = pred[t][rk]
                    if v == t:
                        if rk == 0:
                            acc1 += 1.0
                        if rk < 5:
                            acc5 += 1.0
                        if rk < 10:
                            acc10 += 1.0
                        break
            print(acc1 / tot, acc5 / tot, acc10 / tot)

            samples = sample(valid_idx, 10)
            for t in samples:
                labels = [valid_x1_label[t]]
                for rk in range(10):
                    labels.append(valid_x2_label[pred[t][rk]])
                print(' '.join(labels))
