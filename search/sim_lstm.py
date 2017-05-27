import tensorflow as tf

class SimLSTM(object):
    def lstm_rnn(self, x, dropout, num_layers, size, seq_length, scope):
        """
        Lstm RNN

        :param x: data shape (batch_size, n_steps, n_input)
        :param dropout:
        :param num_layers:
        :param size:
        :param seq_length:
        :return:
        """

        # input format: batch_size, n_steps, n_input (batch_size, seq_length, embedding_size)
        # output format: n_step tensors list of shape batch_size, n_input (seq_length, embedding_size)
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, size])
        x = tf.split(0, seq_length, x)
        print x

        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(seq_length, forget_bias=1.0, state_is_tuple=True)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=dropout)
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * num_layers, state_is_tuple=True)
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(seq_length, forget_bias=1.0, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=dropout)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * num_layers, state_is_tuple=True)
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
        return outputs[-1]

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size

    def __init__(self, seq_length, vocab_size, size,
                 dropout, num_layers,
                 batch_size):
        self.input_x1 = tf.placeholder(tf.int32, [None, seq_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, seq_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")

        self.dropout = tf.Variable(float(dropout), trainable=False)

        # Embedding layer
        self.W = tf.Variable(tf.random_uniform([vocab_size, size], -1.0, 1.0),
                             trainable=True, name="W")
        self.embedding_x1 = tf.nn.embedding_lookup(self.W, self.input_x1)
        self.embedding_x2 = tf.nn.embedding_lookup(self.W, self.input_x2)

        # Output layer
        self.out1 = self.lstm_rnn(self.embedding_x1, self.dropout, num_layers, size, seq_length, "x1")
        self.out2 = self.lstm_rnn(self.embedding_x2, self.dropout, num_layers, size, seq_length, "x2")
        self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(self.out1, self.out2)), 1, keep_dims=True))
        self.distance = tf.div(self.distance,
                               tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
        self.distance = tf.reshape(self.distance, [-1], name="distance")

        self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)
