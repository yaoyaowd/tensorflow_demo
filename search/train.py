import os
import tensorflow as tf
import numpy as np

from search.input_helper import InputHelper
from search.sim_lstm import SimLSTM
from search.vocabulary_processor import VocabularyProcessor

tf.flags.DEFINE_string("train", "", "training file (default: None)")
tf.flags.DEFINE_string("model", "", "training model file")

CHECK_POINT = 100
BATCH_SIZE = 32
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

max_document_length = 20
input_helper = InputHelper()
if os.path.exists(os.path.join(FLAGS.model, "vocab")):
    print "load existing vocab {}".format(os.path.join(FLAGS.model, "vocab"))
    vocab = VocabularyProcessor(max_document_length)
    vocab = vocab.restore(os.path.join(FLAGS.model, "vocab"))
    print "vocab size {}".format(len(vocab.vocabulary_))
    train_set, vocab = input_helper.load_data(FLAGS.train, max_document_length, vocab)
else:
    train_set, vocab = input_helper.load_data(FLAGS.train, max_document_length)
    vocab.save(os.path.join(FLAGS.model, "vocab"))
    print "vocab size {}".format(len(vocab.vocabulary_))

print "initial graph"
with tf.Session() as sess:
    model = SimLSTM(
        max_document_length,
        len(vocab.vocabulary_),
        128, 0.5, 2, BATCH_SIZE)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(0.001)
    print "initialized model"

    saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model)
    if ckpt:
        print "reading model parameters from {}".format(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    grands_and_vars = optimizer.compute_gradients(model.loss)
    tr_op_set = optimizer.apply_gradients(grands_and_vars, global_step=global_step)
    sess.run(tf.initialize_all_variables())
    print "initialized variables"

    def train_step(x1, x2, y):
        feed_dict = {
            model.input_x1: x1,
            model.input_x2: x2,
            model.input_y: y}
        _, step, loss, dist = sess.run(
            [tr_op_set, global_step, model.loss, model.distance], feed_dict)
        acc = 0
        for i in xrange(len(y)):
            if dist[i] < 0.5 and y[i] == 1:
                acc += 1
            elif dist[i] >= 0.5 and y[i] == 0:
                acc += 1
        return step, loss, acc

    batches = input_helper.batch_iter(
        list(zip(train_set[0], train_set[1], train_set[2])), BATCH_SIZE, 200000)
    agg_loss = 0
    agg_acc = 0
    agg_total = 0
    for batch in batches:
        if len(batch) < 1:
            continue
        x1, x2, y = zip(*batch)
        if len(y) < 1:
            continue

        step, loss, acc = train_step(x1, x2, y)
        agg_loss += loss
        agg_acc += acc
        agg_total += BATCH_SIZE
        if step % CHECK_POINT == 0:
            print "Train step {}, total loss {} acc/total {}/{}".format(step, agg_loss, agg_acc, agg_total)
            checkpoint_path = os.path.join(FLAGS.model, "sim.ckpt")
            saver.save(sess, checkpoint_path, global_step=global_step)
            agg_loss = 0
            agg_acc = 0
            agg_total = 0
