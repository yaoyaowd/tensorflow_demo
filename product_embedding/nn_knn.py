import os
import numpy as np
import tensorflow as tf

product_dict = dict()
id_to_product = dict()
paragraphs = list()

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "Input filename")
flags.DEFINE_string("model", "", "Model filename")


class NNKNNModel:
    def __init__(self, product_size, embedding_size, batch_size):
        self.batch_size = batch_size
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            with tf.device('/cpu:0'):
                embeddings = tf.Variable(tf.random_uniform([product_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

                output_embed = tf.nn.embedding_lookup(embeddings, self.train_labels)

            weights = tf.Variable(tf.random_normal([embedding_size, embedding_size]))
            bias = tf.Variable(tf.random_normal([embedding_size]))
            output_layer = tf.matmul(embed, weights) + bias

            self.loss = tf.reduce_sum(tf.abs(tf.add(output_layer, tf.negative(output_embed))), reduction_indices=1)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(self.loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.normalized_embeddings = embeddings / norm

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
                if step > 0 and step % 20000 == 0:
                    print "Average loss at step: ", self.global_step.eval(), " loss: ", average_loss / 2000
                    average_loss = 0
                    self.final_embeddings = self.normalized_embeddings.eval()

                    checkpoint_path = os.path.join(FLAGS.model, "pe.ckpt")
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


def load_data(filename):
    num_lines_read = 0
    prev_uid = ''
    pids = []
    with open(filename) as f:
        for line in f:
            num_lines_read += 1
            items = line.strip().split('\t')
            if len(items) != 3:
                continue

            uid, action_time, pid = items[0], items[1], items[2]
            if pid not in product_dict:
                product_dict[pid] = len(product_dict) + 1
                id_to_product[len(product_dict)] = pid

            if uid != prev_uid and len(pids) > 0:
                paragraphs.append(pids)
                prev_uid = uid
                pids = []
            pids.append(pid)

            if num_lines_read % 1000000 == 0:
                print "read %d lines" % num_lines_read
                print "saw %d different products" % len(product_dict)
            if num_lines_read % 10000000 == 0:
                break


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

    model = NNKNNModel(len(product_dict) + 1, 128, 128)
    load_data(FLAGS.input)
    model.train(num_steps=2000000, generate_batch=generate_batch)


if __name__ == '__main__':
    main()