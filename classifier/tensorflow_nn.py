import tensorflow.python.platform

import numpy as np
import tensorflow as tf

BATCH_SIZE = 100

tf.app.flags.DEFINE_string('train', None, 'File contains training data')
tf.app.flags.DEFINE_string('test', None, 'File contains the test data')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Number of iterations')
FLAGS = tf.app.flags.FLAGS


# Extract numpy representations of the labels and features given rows consisting of:
# label, feat_0, feat_1, ..., feat_n
def extract_data(filename):
    labels = []
    features = []
    num_labels = 0
    # cnt = 0
    for line in file(filename):
        row = line.strip().split(",")
        label = int(row[0])
        num_labels = max(label + 1, num_labels)
        labels.append(label)
        features.append([float(x) for x in row[1:]])
        # cnt += 1
        # if cnt == 20000:
        #     break

    labels_np = np.array(labels).astype(np.uint32)
    features_np = np.matrix(features).astype(np.float32)
    labels_onehot = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)
    return num_labels, features_np,labels_onehot


def multilayer_preceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


def main(argv=None):
    num_labels, train_data,train_labels = extract_data("/Users/dong/Downloads/query_category_train.csv") # FLAGS.train)
    num_labels, test_data, test_labels = extract_data("/Users/dong/Downloads/query_category.csv") # FLAGS.test)
    train_size, num_features = train_data.shape
    print 'shape', num_labels, train_size, num_features

    x = tf.placeholder("float", shape=[None, num_features])
    y = tf.placeholder("float", shape=[None, num_labels])

    n_hidden_1 = 64
    # n_hidden_2 = 10
    weights = {
        'h1': tf.Variable(tf.random_normal([num_features, n_hidden_1])),
        # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, num_labels]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        # 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_labels]))
    }
    pred = multilayer_preceptron(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        # Train
        sess.run(init)
        for epoch in range(100): # FLAGS.num_epochs):
            avg_cost = 0
            total_batch = train_size // BATCH_SIZE
            for i in range(total_batch):
                offset = i * BATCH_SIZE
                batch_data = train_data[offset : (offset + BATCH_SIZE), :]
                batch_labels = train_labels[offset : (offset + BATCH_SIZE)]
                _, c = sess.run([optimizer, cost],
                                feed_dict={x: batch_data, y: batch_labels})
                avg_cost += c / total_batch
            print 'Epoch:', epoch + 1, 'cost=', avg_cost

            if epoch % 20 == 19:
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y: test_labels})

        # Evaluation.
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y: test_labels})



if __name__ == '__main__':
    tf.app.run()