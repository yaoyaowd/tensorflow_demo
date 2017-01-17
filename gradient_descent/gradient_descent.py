import tensorflow as tf
import numpy as np

x = tf.Variable(0., name='x')
model = tf.initialize_all_variables()

# with tf.Session() as sess:
#     for i in range(5):
#         sess.run(model)
#         x = x + 1
#         print sess.run(x)

# threshold = tf.constant(5.)
#
# with tf.Session() as sess:
#     sess.run(model)
#     while sess.run(tf.less(x, threshold)):
#         x = x + 1
#         x_value = sess.run(x)
#         print x_value

x = tf.placeholder("float")
y = tf.placeholder("float")
w = tf.Variable([1.0, 2.0], name='w')
y_model = tf.mul(x, w[0]) + w[1]
error = tf.square(y - y_model)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

model = tf.initialize_all_variables()

errors = []
with tf.Session() as session:
    session.run(model)

    for i in range(1000):
        x_value = np.random.rand()
        y_value = x_value * 2 + 6
        _, error_value = session.run(
            [train_op, error], feed_dict={x: x_value, y: y_value})
        errors.append(error_value)

    w_value = session.run(w)
    print w_value[0], w_value[1]


import matplotlib.pyplot as plt

plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
plt.show()