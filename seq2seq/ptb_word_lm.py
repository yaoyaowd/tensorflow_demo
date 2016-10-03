import time

import numpy as np
import tensorflow as tf

import reader


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "small", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


class PTBModel(object):
    """The PTB model."""
    def __init__(self, is_training, config):
        self._batch_size = config.batch_size
        self._num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self._input_data = tf.placeholder(tf.int32, [self._batch_size, self._num_steps])
        self._targets = tf.placeholder(tf.int32, [self._batch_size, self._num_steps])

        # Create LSTM cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1.0:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._init_state = cell.zero_state(self._batch_size, tf.float32)

        # Create embeddings for words
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)
        if is_training and config.keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self._init_state
        with tf.variable_scope("RNN"):
            for time_step in range(self._num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_ouptut, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_ouptut)
        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([self._batch_size * self._num_steps])])
        self._cost = tf.reduce_sum(loss) / self._batch_size
        self._final_state = state

        if not is_training:
            return
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self._lr, lr_value))


def run_epoch(session, m, data, eval_op):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m._batch_size) - 1) // m._num_steps
    costs = 0.0
    iters = 0
    state = m._init_state.eval()
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m._batch_size, m._num_steps)):
        # enumerate every step, training inputs x and targets y.
        cost, state, _ = session.run([m._cost, m._final_state, eval_op],
                                     {m._input_data: x, m._targets: y, m._init_state: state})
        costs += cost
        iters += m._num_steps
        if step % 10 == 0:
            print "%.3f perplexity: %.3f " % (step * 1.0 / epoch_size, np.exp(costs / iters))
    return np.exp(costs / iters)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    train_data, valid_data, test_data, _ = reader.ptb_raw_data(FLAGS.data_path)
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)

        tf.initialize_all_variables().run()
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print "Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m._lr))
            train_perplexity = run_epoch(session, m, train_data, m._train_op)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))


class SmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


if __name__ == "__main__":
  tf.app.run()