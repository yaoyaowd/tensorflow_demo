import numpy as np
import tensorflow as tf

BATCH_SIZE = 100

tf.app.flags.DEFINE_string('train', None, 'File contains training data')
tf.app.flags.DEFINE_string('test', None, 'File contains the test data')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Number of iterations')
tf.app.flags.DEFINE_integer('learning_rate', 0.01, 'Number of iterations')
FLAGS = tf.app.flags.FLAGS


# Extract numpy representations of the labels and features given rows consisting of:
# label, feat_0, feat_1, ..., feat_n
def extract_data(filename):
    labels = []
    features = []
    num_labels = 0
    for line in file(filename):
        row = line.split(",")
        label = int(row[0])
        labels.append(label)
        num_labels = max(num_labels, label + 1)
        features.append([float(x) for x in row[1:]])

    labels_np = np.array(labels).astype(np.uint8)
    features_np = np.matrix(features).astype(np.float32)
    labels_onehot = (np.arange(num_labels) == labels_np[:, None]).astype(np.float32)
    return num_labels, features_np, labels_onehot


def main(argv=None):
    train_data,train_labels = extract_data(FLAGS.train)
    test_data, test_labels = extract_data(FLAGS.test)
    num_labels, train_size, num_features = train_data.shape

    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, num_labels])

    # Define and initialize the network.
    W = tf.Variable(tf.zeros([num_features, num_labels]))
    b = tf.Variable(tf.zeros([num_labels]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Optimization.
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    # Evaluation.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create local session to train and test
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        print 'Initialized!'
        tf.initialize_all_variables().run()

        # Iterate and train.
        print 'Training.'
        for step in xrange(FLAGS.num_epochs * train_size // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset : (offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset : (offset + BATCH_SIZE)]
            train_step.run(feed_dict={x: batch_data, y_: batch_labels})
            if step % 100 == 0:
                print 'Step:', step, 'Accuracy:', accuracy.eval(feed_dict={x: batch_data, y_: batch_labels})

        # Give very detailed output.
        print
        print 'Weight matrix.'
        print s.run(W)
        print
        print 'Bias vector.'
        print s.run(b)
        print
        print "Applying model to first test instance."
        first = test_data[:1]
        print "Point =", first
        print "Wx+b = ", s.run(tf.matmul(first, W) + b)
        print "softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(first, W) + b))
        print
        print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})


if __name__ == '__main__':
    tf.app.run()