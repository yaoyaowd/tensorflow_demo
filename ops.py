import tensorflow as tf

def restore_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var

def init_variable(shape, stddev=0.02, name=None):
    init = tf.truncated_normal(shape=shape, stddev=stddev)
    if name is None:
        return tf.Variable(init)
    else:
        return tf.get_variable(name, initalizer=init)

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    """A conv layer of size k_h*k_w*input_.get_shape()[-1] -> output_dim, with stride d_h, d_w."""
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        return conv

def dilated_conv2d(input_, output_dim, k_h=3, k_w=3, dilation=2, stddev=0.02, name="dilated_conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.atrous_conv2d(input_, w, rate=dilation, padding="SAME")
        conv = tf.nn.bias_add(conv, biases)
        return conv

def residual_block(input_, output_dim, k_h=3, k_w=3, name="resid"):
    with tf.variable_scope(name):
        tmp = tf.nn.relu(conv2d(input_, output_dim, k_h=k_h, k_w=k_w, d_h=1, d_w=1, name="c1"))
        return input_ + conv2d(tmp, output_dim, k_h=k_h, k_w=k_w, d_h=1, d_w=1, name="c2")

def conv2d_transpose(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                     name="conv2d_transpose", with_w=False):
    """A conv layer of size k_h*k_w*output_shape[-1] -> input_.get_shape()[-1], with stride d_h, d_w.
    Return deconv network:
        value: 4-D tensor input [batch, height, width, in_channels]
        filter: 4-D tensor [height, weight, output_channels, in_channels]
        output_shape: A 1-D tensor representing the output shape of the deconvolution op
        strides: [1, d_h, d_w, 1]"""
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        if with_w:
            return deconv, w, biases
        return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    """By default, y=0.6*x+0.4*abs(x)"""
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """Create linear transform input_*matrix + bias."""
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def max_pool(input_, k_h=5, k_w=5, d_h=2, d_w=2):
    return tf.nn.max_pool(input_, ksize=[1, k_h, k_w, 1], strides=[1, d_h, d_w, 1], padding="SAME")

def avg_pool(input_, k_h=5, k_w=5, d_h=2, d_w=2):
    return tf.nn.avg_pool(input_, ksize=[1, k_h, k_w, 1], strides=[1, d_h, d_w, 1], padding="SAME")

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            center=True,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)

