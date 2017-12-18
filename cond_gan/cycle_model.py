from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
from cond_gan.pix2pix_util import *

IMAGE_SIZE = 256
BATCH_SIZE = 16
DIM = 64
LEARNING_RATE = 0.0002
BETA1 = 0.5
EPOCH = 200

# https://github.com/xhujoy/CycleGAN-tensorflow

def discriminator(image, reuse=False, name="discriminator"):
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = lrelu(conv2d(image, DIM, k_h=4, k_w=4, name="d_h0"))
        h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, DIM * 2, name="d_h1")))
        h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, DIM * 4, name="d_h2")))
        h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, DIM * 8, name="d_h3")))
        h4 = linear(tf.reshape(h3, [BATCH_SIZE, -1]), 1, "d_h4")
        return h4


def generator(image, reuse=False, name="generator"):
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        e1 = conv2d(image, DIM, name='g_e1_conv')
        e2 = batch_norm(name='g_bn_e2')(conv2d(lrelu(e1), DIM * 2, name='g_e2_conv'))
        e3 = batch_norm(name='g_bn_e3')(conv2d(lrelu(e2), DIM * 4, name='g_e3_conv'))
        e4 = batch_norm(name='g_bn_e4')(conv2d(lrelu(e3), DIM * 8, name='g_e4_conv'))
        e5 = batch_norm(name='g_bn_e5')(conv2d(lrelu(e4), DIM * 8, name='g_e5_conv'))
        e6 = batch_norm(name='g_bn_e6')(conv2d(lrelu(e5), DIM * 8, name='g_e6_conv'))
        e7 = batch_norm(name='g_bn_e7')(conv2d(lrelu(e6), DIM * 8, name='g_e7_conv'))
        e8 = batch_norm(name='g_bn_e8')(conv2d(lrelu(e7), DIM * 8, name='g_e8_conv'))

        d1 = conv2d_transpose(tf.nn.relu(e8), [BATCH_SIZE, 2, 2, DIM * 8], name='g_d1')
        d1 = tf.nn.dropout(batch_norm(name='g_bn_d1')(d1), 0.5)
        d1 = tf.concat([d1, e7], 3)

        d2 = conv2d_transpose(tf.nn.relu(d1), [BATCH_SIZE, 4, 4, DIM * 8], name='g_d2')
        d2 = tf.nn.dropout(batch_norm(name='g_bn_d2')(d2), 0.5)
        d2 = tf.concat([d2, e6], 3)

        d3 = conv2d_transpose(tf.nn.relu(d2), [BATCH_SIZE, 8, 8, DIM * 8], name='g_d3')
        d3 = tf.nn.dropout(batch_norm(name='g_bn_d3')(d3), 0.5)
        d3 = tf.concat([d3, e5], 3)

        d4 = conv2d_transpose(tf.nn.relu(d3), [BATCH_SIZE, 16, 16, DIM * 8], name='g_d4')
        d4 = batch_norm(name='g_bn_d4')(d4)
        d4 = tf.concat([d4, e4], 3)

        d5 = conv2d_transpose(tf.nn.relu(d4), [BATCH_SIZE, 32, 32, DIM * 4], name='g_d5')
        d5 = batch_norm(name='g_bn_d5')(d5)
        d5 = tf.concat([d5, e3], 3)

        d6 = conv2d_transpose(tf.nn.relu(d5), [BATCH_SIZE, 64, 64, DIM * 2], name='g_d6')
        d6 = batch_norm(name='g_bn_d6')(d6)
        d6 = tf.concat([d6, e2], 3)

        d7 = conv2d_transpose(tf.nn.relu(d6), [BATCH_SIZE, 128, 128, DIM], name='g_d7')
        d7 = batch_norm(name='g_bn_d7')(d7)
        d7 = tf.concat([d7, e1], 3)

        d8 = conv2d_transpose(tf.nn.relu(d7), [BATCH_SIZE, 256, 256, 3], name='g_d8')
        return tf.nn.tanh(d8)


class cycleGAN(object):
    def __init__(self,
                 sess,
                 dataset_name='facades',
                 checkpoint_dir=None):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        self.real_data = tf.placeholder(tf.float32,
                                        [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3 + 3],
                                        name='input_images')
        self.real_A = self.real_data[:, :, :, :3]
        self.real_B = self.real_data[:, :, :, 3:6]

        self.fake_B = generator(self.real_A, name="generatorA2B")
        self.fake_A = generator(self.real_B, name="generatorB2A")
        self.fake_B_fake_A = generator(self.fake_B, reuse=True, name="generatorB2A")
        self.fake_A_fake_B = generator(self.fake_A, reuse=True, name="generatorA2B")

        self.DA_real = discriminator(self.real_A, reuse=False, name="descriminatorA")
        self.DB_real = discriminator(self.real_B, reuse=False, name="descriminatorB")
        self.DA_fake = discriminator(self.fake_A, reuse=True, name="descriminatorA")
        self.DB_fake = discriminator(self.fake_B, reuse=True, name="descriminatorB")

        self.g_loss_a2b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.DB_fake, labels=tf.ones_like(self.DB_fake))) + 100 * tf.reduce_mean(
            tf.abs(self.real_A - self.fake_B_fake_A)) + 100 * tf.reduce_mean(
            tf.abs(self.real_B - self.fake_B))
        self.g_loss_b2a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.DA_fake, labels=tf.ones_like(self.DA_fake))) + 100 * tf.reduce_mean(
            tf.abs(self.real_B - self.fake_A_fake_B)) + 100 * tf.reduce_mean(
            tf.abs(self.real_A - self.fake_A))
        self.g_loss = self.g_loss_a2b + self.g_loss_b2a

        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.DB_fake, labels=tf.zeros_like(self.DB_fake))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.DB_real, labels=tf.ones_like(self.DB_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.DA_fake, labels=tf.zeros_like(self.DA_fake))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.DA_real, labels=tf.ones_like(self.DA_real)))

        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.real_A_sum = tf.summary.image("real_A", self.real_A)
        self.real_B_sum = tf.summary.image("real_B", self.real_B)
        self.fake_A_sum = tf.summary.image("fake_A", self.fake_A)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)
        self.fake_AB_sum = tf.summary.image("fake_AB", self.fake_A_fake_B)
        self.fake_BA_sum = tf.summary.image("fake_BA", self.fake_B_fake_A)

        self.d_sum = tf.summary.merge([self.d_loss_sum])
        self.g_sum = tf.summary.merge([self.g_loss_sum, self.g_loss_a2b_sum, self.g_loss_b2a_sum,
                                       self.real_A_sum, self.real_B_sum, self.fake_A_sum,
                                       self.fake_B_sum, self.fake_AB_sum, self.fake_BA_sum])

        training_vars = tf.trainable_variables()
        self.d_vars = [var for var in training_vars if 'd_' in var.name]
        self.g_vars = [var for var in training_vars if 'g_' in var.name]
        self.saver = tf.train.Saver(max_to_keep=5)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def train(self):
        d_opt = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(self.d_loss, var_list=self.d_vars)
        g_opt = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(self.g_loss, var_list=self.g_vars)

        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.checkpoint_dir + "logs", self.sess.graph)

        if self.load(self.checkpoint_dir):
            print(" [*] Load success")
        else:
            print(" [!] Load failed, new model...")

        counter = 1
        start_time = time.time()

        for ep in xrange(EPOCH):
            data = glob(self.dataset_name)
            batches = len(data) // BATCH_SIZE
            print("Epoch: {}, num files: {}".format(EPOCH, len(data)))
            for i in xrange(0, batches):
                batch_files = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                batch = [load_data(f) for f in batch_files]
                images = np.array(batch).astype(np.float32)

                _, summary_str, errD = self.sess.run(
                    [d_opt, self.d_sum, self.d_loss],
                    feed_dict={self.real_data: images})
                self.writer.add_summary(summary_str, counter)
                _, summary_str = self.sess.run([g_opt, self.g_sum],
                                               feed_dict={self.real_data: images})
                self.writer.add_summary(summary_str, counter)
                _, summary_str, errG = self.sess.run([g_opt, self.g_sum, self.g_loss],
                                               feed_dict={self.real_data: images})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print("Epoch: {} {}/{} time: {} d_loss: {} g_loss: {}".format(
                    ep, i, batches, time.time() - start_time, errD, errG))

                if np.mod(counter, 100) == 0:
                    if not os.path.exists(self.checkpoint_dir):
                        os.makedirs(self.checkpoint_dir)
                    self.saver.save(self.sess, self.checkpoint_dir, global_step=counter)
