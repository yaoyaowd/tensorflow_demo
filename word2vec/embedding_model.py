import math

import tensorflow as tf


class EmbeddingModel:

    def __init__(self, vocabulary_size, embedding_size, batch_size):
        self.batch_size = batch_size
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            with tf.device('/cpu:0'):
                # Create initial embeddings for each words and look up helper function.
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                # Create a look up table to look up vector for input.
                embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
                # normalize weights for word embedding matrics
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                              stddev = 1.0 / math.sqrt(embedding_size)))
                nce_bias = tf.Variable(tf.zeros([vocabulary_size]))

            self.loss = tf.reduce_mean(tf.nn.nce_loss(
                nce_weights, nce_bias, embed, self.train_labels, batch_size / 2, vocabulary_size))
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(self.loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.normalized_embeddings = embeddings / norm

            self.init = tf.initialize_all_variables()

    def train(self, num_steps, generate_batch):
        with tf.Session(graph=self.graph) as session:
            self.init.run()
            print "Initialized"

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = generate_batch(self.batch_size)
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val
                if step > 0 and step % 2000 == 0:
                    print "Average loss at step: ", step, " loss: ", average_loss / 2000
                    average_loss = 0
            self.final_embeddings = self.normalized_embeddings.eval()