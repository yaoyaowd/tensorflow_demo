from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = "notMNIST.pickle"

with open(pickle_file, 'rb') as f:
    save =pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 128
hidden_dimension = 128
graph = tf.Graph()

# A relu nn softmax cross entropy model for notMNIST:
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    l1_weights = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_dimension]))
    l1_bias = tf.Variable(tf.zeros([hidden_dimension]))
    l1_output = tf.nn.relu(tf.add(tf.matmul(tf_train_dataset, l1_weights), l1_bias))
    l2_weights = tf.Variable(tf.truncated_normal([hidden_dimension, num_labels]))
    l2_bias = tf.Variable(tf.zeros([num_labels]))
    l2_output = tf.add(tf.matmul(l1_output, l2_weights), l2_bias)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l2_output, labels=tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    train_prediction = tf.nn.softmax(l2_output)
    valid_l1_output = tf.nn.relu(tf.add(tf.matmul(tf_valid_dataset, l1_weights), l1_bias))
    valid_l2_output = tf.add(tf.matmul(valid_l1_output, l2_weights), l2_bias)
    valid_prediction = tf.nn.softmax(valid_l2_output)
    test_l1_output = tf.nn.relu(tf.add(tf.matmul(tf_test_dataset, l1_weights), l1_bias))
    test_l2_output = tf.add(tf.matmul(test_l1_output, l2_weights), l2_bias)
    test_prediction = tf.nn.softmax(test_l2_output)

"""
# A linear softmax cross entropy model for notMNIST:
with graph.as_default():
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
"""

num_steps = 100001

def accuracy(predictions, labels):
    return np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / float(predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(1, num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %f" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %f" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %f" % accuracy(test_prediction.eval(), test_labels))
