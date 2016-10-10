import data_utils
import math
import os
import sys

import numpy as np
import tensorflow as tf

from seq2seq import seq2seq_model


LEARNING_RATE = 0.5
LEARNING_RATE_DECAY_FACTOR = 0.99
MAX_GRADIENT_NORM = 5.0
BATCH_SIZE = 64
SIZE = 512
NUM_LAYERS = 2
EN_VOCAB_SIZE = 31
STEPS_PER_CHECKPOINT = 100
MAX_LENGTH = 22
DATA_PATH = "/Users/dong/Downloads/navboost/"

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(22, 22)] #, (30, 32), (50, 50)]


def prepare_spell_correction(input_file):
    vocabulary = data_utils.initialize_a_to_z()
    train_set = [[] for _ in _buckets]
    line_count = 0
    with open(input_file) as f:
        for line in f.readlines():
            items = line.strip().lower().split('\t')
            typo_ids = data_utils.string_to_token_ids(items[1], vocabulary)
            correction_ids = data_utils.string_to_token_ids(items[2], vocabulary)
            correction_ids.append(data_utils.EOS_ID)
            if (data_utils.UNK_ID) in typo_ids or (len(typo_ids) > MAX_LENGTH):
                continue
            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(typo_ids) < source_size and len(correction_ids) < target_size:
                    train_set[bucket_id].append([typo_ids, correction_ids])
                    break
            line_count += 1
            if line_count % 10000 == 0:
                print "load lines", line_count
    for bucket_id, _ in enumerate(_buckets):
        print "Train set sample:", train_set[bucket_id][:10]
    return train_set, train_set, vocabulary


def create_model(session, forward_only):
    model = seq2seq_model.Seq2SeqModel(EN_VOCAB_SIZE, EN_VOCAB_SIZE, _buckets,
                                       SIZE, NUM_LAYERS, MAX_GRADIENT_NORM, BATCH_SIZE,
                                       LEARNING_RATE, LEARNING_RATE_DECAY_FACTOR,
                                       forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(DATA_PATH)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print "created model with fresh parameters"
        session.run(tf.initialize_all_variables())
    return model


def train():
    train_set, dev_set, vocabulary = prepare_spell_correction(DATA_PATH + "spell_correction.txt")
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    print "total size:", train_total_size
    print "data percentage:", train_buckets_scale

    with tf.Session() as session:
        model = create_model(session, False)

        loss = 0.0
        current_step = 0
        previous_losses = []
        while current_step < 10000000:
            current_step += 1

            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_bucket_sizes[i] > random_number_01])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

            _, step_loss, _ = model.step(
                session, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
            loss += step_loss / STEPS_PER_CHECKPOINT

            if current_step % STEPS_PER_CHECKPOINT == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print "global step %d learning rate %.4f perpexity %.2f" % (
                    model.global_step.eval(), model.learning_rate.eval(), perplexity)
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    session.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                checkpoint_path = os.path.join(DATA_PATH, "spell_correction.ckpt")
                model.saver.save(session, checkpoint_path, global_step=model.global_step)

                loss = 0.0
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print "eval: empty bucket %d" % bucket_id
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
                    _, eval_loss, _ = model.step(
                        session, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print "eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx)
                    sys.stdout.flush()

                decode(session, model, vocabulary, dev_set)


def visualize(rev_vocabulary, inputs, should_reverse=False):
    if isinstance(inputs, list):
        output_matrix = np.empty((len(inputs), BATCH_SIZE))
        for i in xrange(len(inputs)):
            for j in xrange(len(inputs[i])):
                output_matrix[i][j] = inputs[i][j]
        inputs = output_matrix

    inputs = inputs.transpose()
    ret = []
    for input in inputs:
        str = ""
        for c in input:
            if c == data_utils.EOS_ID:
                break
            str = str + rev_vocabulary.get(c)
        if should_reverse:
            str = str[::-1]
        ret.append(str)
    return ret


def decode(sess, model, vocabulary, dev_set):
    rev_vocabulary = {v: k for k, v in vocabulary.items()}
    rev_vocabulary[data_utils.PAD_ID] = ''
    rev_vocabulary[data_utils.GO_ID] = ''
    rev_vocabulary[data_utils.UNK_ID] = ''
    for bucket_id in xrange(len(dev_set)):
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)

        output_matrix = np.empty((len(output_logits), BATCH_SIZE))
        for lenIdx in xrange(len(output_logits)):
            output_matrix[lenIdx] = np.array([int(np.argmax(logit)) for logit in output_logits[lenIdx]])

        # print encoder_inputs
        # print  decoder_inputs
        typos = visualize(rev_vocabulary, encoder_inputs, should_reverse=True)
        corrections = visualize(rev_vocabulary, decoder_inputs)
        guesses = visualize(rev_vocabulary, output_matrix)
        total = 0
        corrected = 0
        for i in xrange(len(typos)):
            total += 1
            corrected += 1 if corrections[i] == guesses[i] else 0
            print typos[i], ' - ', corrections[i], ' - ', guesses[i]
        print 'total: ', total, ' corrected: ', corrected, ' acc: ', corrected / (total + 0.0)


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
