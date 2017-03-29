import math
import numpy as np
import os
import sys
import tensorflow as tf


EMBEDDING_SIZE = 128
BATCH_SIZE = 128
NUM_STEPS = 20000001
MAX_PRODUCT_SIZE = 2400000

product_dict = dict()
id_to_product = dict()
paragraphs = list()


class EmbeddingModel:
    def __init__(self, product_size, embedding_size, batch_size):
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
                    [product_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
                nce_bias = tf.Variable(tf.zeros([product_size]))

            self.loss = tf.reduce_mean(tf.nn.nce_loss(
                nce_weights, nce_bias, embed, self.train_labels, batch_size / 2, product_size))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(
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
            ckpt = tf.train.get_checkpoint_state('/Users/dwang/Downloads/pe')
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
                if step > 0 and step % 20000 == 0:
                    print "Average loss at step: ", self.global_step.eval(), " loss: ", average_loss / 2000
                    average_loss = 0
                    self.final_embeddings = self.normalized_embeddings.eval()

                    checkpoint_path = os.path.join(
                        '/Users/dwang/Downloads/pe', "pe.ckpt")
                    self.saver.save(session, checkpoint_path, global_step=self.global_step)

                    if step % 100000 == 0:
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


def process(actions, pids):
    if ('1005' in actions or '1267' in actions) and len(actions) > 1:
        count = 0
        for i in actions:
            count += 1 if i == '1267' else 0
        if count <= 1:
            return 0

        vpids = []
        for i in range(len(pids)):
            if actions[i] == '1267':
                if pids[i] not in product_dict:
                    product_dict[pids[i]] = len(product_dict) + 1
                    id_to_product[len(product_dict)] = pids[i]
                vpids.append(product_dict[pids[i]])
        paragraphs.append(vpids)
        return 1

    return 0


def load_data(filename):
    num_lines_read = 0
    num_users_valid_actions_seq = 0
    prev_uid = ''
    actions = []
    pids = []

    with open(filename) as f:
        for line in f:
            num_lines_read += 1
            items = line.strip().split('\t')
            if len(items) != 4:
                continue

            uid = items[0]
            action = items[2]
            pid = items[3]
            if uid != prev_uid:
                num_users_valid_actions_seq += process(actions, pids)
                prev_uid = uid
                actions = []
                pids = []
            actions.append(action)
            pids.append(pid)

            if num_lines_read % 1000000 == 0:
                print "read %d lines" % num_lines_read
                print "saw %d valid user action seq" % num_users_valid_actions_seq
                print "saw %d different products" % len(product_dict)
                if num_users_valid_actions_seq > 0:
                    print "last user session:"
                    print paragraphs[-1]

        num_users_valid_actions_seq += process(actions, pids)


data_index = 0
product_index = 0
def generate_batch(batch_size):
    global data_index
    global product_index

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    i = 0
    while i < batch_size:
        product_id = paragraphs[data_index][product_index]
        found = False
        for t in range(product_index + 1, len(paragraphs[data_index])):
            if paragraphs[data_index][t]:
                batch[i] = product_id
                labels[i, 0] = paragraphs[data_index][t]
                i += 1
                found = True
                break

        if found:
            product_index += 1

        if not found or product_index >= len(paragraphs[data_index]) - 1:
            data_index = (data_index + 1) % len(paragraphs)
            product_index = 0

    return batch, labels


def main():
    global product_dict
    global paragraphs

    load_data(sys.argv[1])

    model = EmbeddingModel(len(product_dict) + 1, EMBEDDING_SIZE, BATCH_SIZE)
    model.train(num_steps=NUM_STEPS, generate_batch=generate_batch)


if __name__ == '__main__':
    main()