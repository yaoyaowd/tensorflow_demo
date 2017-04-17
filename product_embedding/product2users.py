"""
Inspired by skip gram algorithm.
"""
import random
import math
import numpy as np
import os
import tensorflow as tf


EMBEDDING_SIZE = 128
BATCH_SIZE = 256
NUM_STEPS = 20000001
MAX_PRODUCT_SIZE = 2000000
DISPLAY_EPOCH = 20000
EVAL_EPOCH = 100000
M1 = 1000000
MAX_LINE = M1 * 10

user_dict = dict()
user_freq = dict()
product_dict = dict()
id_to_product = dict()
training_samples = []


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "Input filename")
flags.DEFINE_string("model", "", "Model filename")


class EmbeddingModel:
    def __init__(self, product_size, user_size, embedding_size, batch_size):
        self.batch_size = batch_size
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            self.global_step = tf.Variable(0, trainable=False)

            with tf.device('/cpu:0'):
                embeddings = tf.Variable(tf.random_uniform([product_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
                nce_weights = tf.Variable(tf.truncated_normal(
                    [user_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
                nce_bias = tf.Variable(tf.zeros([user_size]))

            self.loss = tf.reduce_mean(tf.nn.nce_loss(
                nce_weights, nce_bias, embed, self.train_labels, batch_size / 2, user_size))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(
                self.loss, global_step=self.global_step)

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.normalized_embeddings = embeddings / norm

            self.valid_examples = np.random.choice(id_to_product.keys(), 10)
            valid_data = tf.constant(self.valid_examples, dtype=tf.int32)
            valid_embedding = tf.nn.embedding_lookup(embeddings, valid_data)
            self.similarity = tf.matmul(valid_embedding, self.normalized_embeddings, transpose_b=True)

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
                batch_inputs, batch_labels = generate_batch(self.batch_size)
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)

                average_loss += loss_val
                if step > 0 and step % DISPLAY_EPOCH == 0:
                    print "Average loss at step: ", self.global_step.eval(), " loss: ", average_loss / DISPLAY_EPOCH
                    average_loss = 0
                    self.final_embeddings = self.normalized_embeddings.eval()

                    checkpoint_path = os.path.join(FLAGS.model, "pe.ckpt")
                    self.saver.save(session, checkpoint_path, global_step=self.global_step)

                    if step % EVAL_EPOCH == 0:
                        sim = self.similarity.eval()
                        for i in xrange(len(self.valid_examples)):
                            valid_product = id_to_product[self.valid_examples[i]]
                            top_k = 8
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log_str = "Nearest to %s:" % valid_product
                            for k in xrange(top_k):
                                close_word = id_to_product[nearest[k]]
                                log_str = "%s %s," % (log_str, close_word)
                            print log_str


def load_data(filename):
    num_lines_read = 0
    with open(filename) as f:
        for line in f:
            num_lines_read += 1
            items = line.strip().split('\t')
            if len(items) != 2:
                continue

            uid, pid = items[0], items[1]
            if uid not in user_freq:
                user_freq[uid] = 0
            user_freq[uid] = user_freq[uid] + 1
            if num_lines_read % M1 == 0:
                print "read %d lines" % num_lines_read
            if num_lines_read >= MAX_LINE:
                break

    num_lines_read = 0
    with open(filename) as f:
        for line in f:
            num_lines_read += 1
            items = line.strip().split('\t')
            if len(items) != 2:
                continue

            uid, pid = items[0], items[1]
            if user_freq[uid] == 1:
                continue

            if pid not in product_dict:
                product_dict[pid] = len(product_dict) + 1
                id_to_product[len(product_dict)] = pid
            if uid not in user_dict:
                user_dict[uid] = len(user_dict) + 1
            training_samples.append([product_dict[pid], user_dict[uid]])

            if num_lines_read % M1 == 0:
                print "read %d lines" % num_lines_read
                print "saw %d products %d users" % (len(product_dict), len(user_dict))
            if num_lines_read >= MAX_LINE:
                break


def generate_batch(batch_size):
    global training_samples

    batchs = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    samples = random.sample(training_samples, batch_size)
    for i in range(len(samples)):
        batchs[i] = samples[i][0]
        labels[i, 0] = samples[i][1]

    return batchs, labels


def main():
    global product_dict
    global paragraphs

    load_data(FLAGS.input)
    model = EmbeddingModel(
        len(product_dict) + 1, len(user_dict) + 1, EMBEDDING_SIZE, BATCH_SIZE)
    model.train(num_steps=NUM_STEPS, generate_batch=generate_batch)


if __name__ == '__main__':
    main()
