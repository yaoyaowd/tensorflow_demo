import argparse
import os
import sys

import tensorflow as tf
import numpy as np


FEATURE_LIMIT = 87005
BATCH_SIZE = 100
K = 100

BASE = 5               # ps, ws, as, cos
VECTOR_LENGTH = 180     # cos_0, .. cos_VECTOR_LENGTH
INDEX2VECTOR = ['cos_' + str(i) for i in range(VECTOR_LENGTH)]
NUM_TRUE_TAGS = 1054    # s_tt_0, .., s_tt_NUM_TRUE_TAGS
                        # t_tt_0, .., t_tt_NUM_TRUE_TAGS
NUM_WORDS = 50000       # w_0, .., w_NUM_WORDS

FLAGS = None


# Extract numpy representations of the labels and features given rows consisting of:
# label column_id:feature column_id:feature ...
def extract_data(filename, feature_limit):
    print 'load data from:', filename
    labels = []
    features = []
    num_labels = 0
    max_feature = 0
    line_count = 0
    for line in file(filename):
        row = line.split(" ")
        label = int(row[0])
        labels.append(label)
        num_labels = max(num_labels, label + 1)
        example = []
        for item in row[1:]:
            kv = item.split(':')
            k = int(kv[0])
            if k >= FEATURE_LIMIT:
                continue
            v = float(kv[1])
            example.append((k, v))
            max_feature = max(k + 1, max_feature)
        features.append(example)
        line_count += 1
        if line_count % 10000 == 0:
            print 'load %d lines' % line_count

    labels_np = np.array(labels).astype(np.uint8)
    labels_onehot = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)
    return num_labels, max_feature, features, labels_onehot


def transform(data, feature_size):
    features = []
    for d in data:
        example = [0] * feature_size
        for i in d:
            example[i[0]] = i[1]
        features.append(example)
    return np.matrix(features).astype(np.float32)


class LinearModel:
    def __init__(self, num_features, num_labels, learning_rate):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Define and initialize the network.
            self.x = tf.placeholder("float", shape=[None, num_features])
            self.y_ = tf.placeholder("float", shape=[None, num_labels])
            self.W = tf.Variable(tf.zeros([num_features, num_labels]))
            self.b = tf.Variable(tf.zeros([num_labels]))
            self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

            self.global_step = tf.Variable(0, trainable=False)
            self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate) \
                .minimize(self.cross_entropy, global_step=self.global_step)
            self.saver = tf.train.Saver()
            self.init = tf.initialize_all_variables()

            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.correct_sum = tf.reduce_sum(tf.cast(self.correct_prediction, "float"))


def main(_):
    if FLAGS.train_data:
        num_labels, num_features, train_data, train_labels = extract_data(FLAGS.train_data, feature_limit=FEATURE_LIMIT)
    else:
        num_labels, num_features = 2, FEATURE_LIMIT
        train_data, train_labels = [], []
    print "labels", num_labels, "features", num_features

    if FLAGS.test_data:
        _, _, test_data, test_labels = extract_data(FLAGS.test_data, feature_limit=FEATURE_LIMIT)
    else:
        test_data, test_labels = [], []

    train_size = len(train_data)
    model = LinearModel(num_features, num_labels, FLAGS.learning_rate)

    # Create local session to train and test
    with tf.Session(graph=model.graph) as s:
        ckpt = tf.train.get_checkpoint_state(FLAGS.models)
        if ckpt and ckpt.model_checkpoint_path:
            model.saver.restore(s, ckpt.model_checkpoint_path)
            print "Model loaded from", ckpt.model_checkpoint_path
        else:
            model.init.run()
            print "Initialized"

        if test_data:
            print 'testing'
            correct = 0
            total = 0
            tf.initialize_local_variables()
            for i in range(len(test_data) // BATCH_SIZE):
                offset = i * BATCH_SIZE
                batch_data = transform(test_data[offset:(offset + BATCH_SIZE)], num_features)
                batch_labels = test_labels[offset:(offset + BATCH_SIZE)]
                c = s.run(
                    [model.correct_sum],
                    feed_dict={model.x: batch_data, model.y_: batch_labels})
                correct += c[0]
                total += BATCH_SIZE
                print correct, total, "accuracy:", float(correct) / total
            return

        # Iterate and train.
        average_loss = 0
        for step in xrange(FLAGS.train_steps * len(train_data) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % train_size
            batch_data = transform(train_data[offset: (offset + BATCH_SIZE)], num_features)
            batch_labels = train_labels[offset: (offset + BATCH_SIZE)]
            _, loss_val = s.run([model.optimizer, model.cross_entropy],
                                feed_dict={model.x: batch_data, model.y_: batch_labels})
            average_loss += loss_val
            if step > 0 and step % K == 0:
                print "Average loss at step: ", model.global_step.eval(), " loss: ", average_loss / K
                average_loss = 0
                checkpoint_path = os.path.join(FLAGS.models, "pe.ckpt")
                model.saver.save(s, checkpoint_path, global_step=model.global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="")
    parser.add_argument("--train_data", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--train_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=int, default=0.001)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
