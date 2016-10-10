import random

import numpy as np
import tensorflow as tf

from data_utils import PAD_ID, GO_ID


class Seq2SeqModel(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets,
                 size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor,
                 use_lstm=False, num_samples=512, forward_only=False):
        """
        Creat the model.
        :param source_vocab_size:
        :param target_vocab_size:
        :param buckets: a list of pairs (I, O) specifies maximum input length and maximum output length.
        :param size: the size of each layer of the model.
        :param num_layers: the number of layers in the model.
        :param max_gradient_norm:
        :param batch_size:
        :param learning_rate:
        :param learning_rate_decay_factor:
        :param use_lstm:
        :param num_samples: number of samples for sampled softmax.
        :param forward_only:
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # For softmax, we need an output projection to map vector of size m to words.
        output_projection = None
        softmax_loss_function = None
        if num_samples > 0 and num_samples < self.target_vocab_size:
            with tf.device("cpu:/0"):
                w = tf.get_variable("proj_w", [size, self.target_vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                with tf.device("cpu:/0"):
                    labels = tf.reshape(labels, [-1, -1])
                    return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.target_vocab_size)
            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for RNN
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(size) if use_lstm else tf.nn.rnn_cell.GRUCell(size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # To use embeddings for inputs and outputs
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=source_vocab_size, num_decoder_symbols=target_vocab_size,
                embedding_size=size, output_projection=output_projection,
                feed_previous=do_decode)

        # Feeds for inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][0] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses
        if forward_only:
            self.output, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs]
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        # Calculate the gradients and apply it
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
        self.saver = tf.train.Saver(tf.all_variables())


    def get_batch(self, data, bucket_id):
        """
        Get a random batch of data from the specified bucket, prepare for step.
        :param data: a tuple from bucket to a list of data.
        :param bucket_id: an integer
        :return:
        """
        encoder_size, decoder_size = self.buckets[bucket_id]

        # Encoder inputs, decoder inputs are of format [[input1, input2]], [[output1], [output2]]
        encoder_inputs, decoder_inputs = [], []
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input + [PAD_ID] * decoder_pad_size)

        # Transform input, output to batches
        # input -> [[input1 token1, input2 token1, ...], [input1 token2, input2 token2, ...], ...]
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights


    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        """
        Run a step of the model feeding the given inputs
        :param session:
        :param encoder_inputs:
        :param decoder_inputs:
        :param target_weights:
        :param bucket_id:
        :param forward_only:
        :return:
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updates[bucket_id],
                           self.gradient_norms[bucket_id],
                           self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None # Gradient norm, loss, no outputs
        else:
            return None, outputs[0], outputs[1:] # No gradient norm, loss, outputs
