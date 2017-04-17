"""
Input words and output products.
"""
import os
import random
import numpy as np
import tensorflow as tf


LEARNING_RATE = 0.002
TRAINING_EPOCH = 1500000
WORD_SIZE = 50001
EMBEDDING_SIZE = 180
BATCH_SIZE = 32

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "Input filename")
flags.DEFINE_string("model", "", "Model filename")


class NNModel:
    def __init__(self, product_size, word_size, embedding_size, batch_size):
        self.batch_size = batch_size
        self.product_size = product_size
        self.word_size = word_size
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            self.global_step = tf.Variable(0, trainable=False)

            self.train_inputs = tf.placeholder(tf.float32, shape=[None, word_size])
            self.train_labels = tf.placeholder(tf.float32, shape=[None, product_size])

            h1 = tf.Variable(tf.random_normal([word_size, embedding_size]))
            layer1 = tf.nn.relu(tf.matmul(self.train_inputs, h1))
            product_embeddings = tf.Variable(tf.random_uniform(
                [embedding_size, product_size], -1.0, 1.0))
            self.out_layer = tf.matmul(layer1, product_embeddings)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                self.out_layer, self.train_labels))
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=LEARNING_RATE).minimize(self.loss, global_step=self.global_step)

            self.saver = tf.train.Saver()
            self.init = tf.initialize_all_variables()

    def train(self, num_steps, generate_batch):
        with tf.Session(graph=self.graph) as session:
            ckpt = tf.train.get_checkpoint_state(FLAGS.model)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(session, ckpt.model_checkpoint_path)
                print "Model loaded from", ckpt.model_checkpoint_path
            else:
                self.init.run()
                print "Initialized"

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = generate_batch(
                    self.batch_size, self.word_size, self.product_size)
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step > 0 and step % 20 == 0:
                    print "Average loss at step: ", self.global_step.eval(), " loss: ", average_loss / 20
                    average_loss = 0

                    correct_prediction = tf.equal(tf.argmax(self.out_layer, 1), tf.argmax(batch_labels, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print "Accuracy:", accuracy.eval(feed_dict={self.train_inputs: batch_inputs,
                                                                self.train_labels: batch_labels})

                    checkpoint_path = os.path.join(FLAGS.model, "pe.ckpt")
                    self.saver.save(session, checkpoint_path, global_step=self.global_step)


product_dict = dict()
id_to_product = dict()
train_examples = []
labels = []

def load_data(filename):
    num_lines_read = 0
    with open(filename) as f:
        for line in f:
            num_lines_read += 1
            items = line.strip().split('#')
            items = items[0].split('\t')
            pid = items[0]
            words = [int(i) for i in items[1:]]

            id = len(product_dict)
            product_dict[pid] = id
            id_to_product[id] = pid
            train_examples.append(words)
            labels.append(id)


def generate_batch(batch_size, word_size, product_size):
    global labels
    global train_examples

    ids = random.sample(labels, batch_size)
    batch = np.ndarray(shape=(batch_size, word_size), dtype=np.float32)
    labels_onehot = np.ndarray(shape=(batch_size,product_size), dtype=np.float32)
    for i in range(len(ids)):
        labels_onehot[i, ids[i]] = 1
        batch[i, 0] = 1
        for w in train_examples[ids[i]]:
            batch[i, w] = 1
    return batch, labels_onehot


def main():
    load_data(FLAGS.input)
    model = NNModel(len(product_dict), WORD_SIZE, EMBEDDING_SIZE, BATCH_SIZE)
    model.train(num_steps=TRAINING_EPOCH, generate_batch=generate_batch)


if __name__ == '__main__':
    main()
