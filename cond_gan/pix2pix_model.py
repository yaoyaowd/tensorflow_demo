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

# https://github.com/yenchenlin/pix2pix-tensorflow

class pix2pix(object):
    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            h0 = lrelu(conv2d(image, DIM, name="d_h0"))
            h1 = lrelu(self.d_bn1(conv2d(h0, DIM*2, name="d_h1")))
            h2 = lrelu(self.d_bn2(conv2d(h1, DIM*4, name="d_h2")))
            h3 = lrelu(self.d_bn3(conv2d(h2, DIM*8, name="d_h3")))
            h4 = linear(tf.reshape(h3, [BATCH_SIZE, -1]), 1, "d_h4")
            return tf.nn.sigmoid(h4), h4

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:
            # image is 256*256*3, encode image
            e1 = conv2d(image, DIM, name='g_e1_conv')
            e2 = self.g_bn_e2(conv2d(lrelu(e1), DIM*2, name='g_e2_conv'))
            e3 = self.g_bn_e3(conv2d(lrelu(e2), DIM*4, name='g_e3_conv'))
            e4 = self.g_bn_e4(conv2d(lrelu(e3), DIM*8, name='g_e4_conv'))
            e5 = self.g_bn_e5(conv2d(lrelu(e4), DIM*8, name='g_e5_conv'))
            e6 = self.g_bn_e6(conv2d(lrelu(e5), DIM*8, name='g_e6_conv'))
            e7 = self.g_bn_e7(conv2d(lrelu(e6), DIM*8, name='g_e7_conv'))
            e8 = self.g_bn_e8(conv2d(lrelu(e7), DIM*8, name='g_e8_conv'))

            self.d1 = conv2d_transpose(tf.nn.relu(e8), [BATCH_SIZE, 2, 2, DIM*8], name='g_d1')
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)

            self.d2 = conv2d_transpose(tf.nn.relu(d1), [BATCH_SIZE, 4, 4, DIM * 8], name='g_d2')
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)

            self.d3 = conv2d_transpose(tf.nn.relu(d2), [BATCH_SIZE, 8, 8, DIM * 8], name='g_d3')
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)

            self.d4 = conv2d_transpose(tf.nn.relu(d3), [BATCH_SIZE, 16, 16, DIM * 8], name='g_d4')
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)

            self.d5 = conv2d_transpose(tf.nn.relu(d4), [BATCH_SIZE, 32, 32, DIM * 4], name='g_d5')
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)

            self.d6 = conv2d_transpose(tf.nn.relu(d5), [BATCH_SIZE, 64, 64, DIM * 2], name='g_d6')
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)

            self.d7 = conv2d_transpose(tf.nn.relu(d6), [BATCH_SIZE, 128, 128, DIM], name='g_d7')
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)

            self.d8 = conv2d_transpose(tf.nn.relu(d7), [BATCH_SIZE, 256, 256, 3], name='g_d8')
            return tf.nn.tanh(self.d8)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def __init__(self,
                 sess,
                 dataset_name='facades',
                 checkpoint_dir=None):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.real_data = tf.placeholder(tf.float32,
                                        [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3 + 3],
                                        name='input_images')
        self.real_A = self.real_data[:, :, :, :3]
        self.real_B = self.real_data[:, :, :, 3:6]
        self.fake_B = self.generator(self.real_A)
        self.real_A_img = tf.summary.image("real_A_img", self.real_A)
        self.real_B_img = tf.summary.image("real_B_img", self.real_B)
        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)

        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits_, labels=tf.ones_like(self.D_))) + 100 * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_sum = tf.summary.merge([self.d_loss_sum, self.d_loss_real_sum, self.d_loss_fake_sum])
        self.g_sum = tf.summary.merge([
            self.real_A_img, self.real_B_img, self.d__sum, self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])

        training_vars = tf.trainable_variables()
        self.d_vars = [var for var in training_vars if 'd_' in var.name]
        self.g_vars = [var for var in training_vars if 'g_' in var.name]
        self.saver = tf.train.Saver(max_to_keep=5)

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

                _, summary_str, errD_real, errD_fake = self.sess.run(
                    [d_opt, self.d_sum, self.d_loss_real, self.d_loss_fake],
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
                    ep, i, batches, time.time() - start_time, errD_real+errD_fake, errG))

                if np.mod(counter, 100) == 0:
                    if not os.path.exists(self.checkpoint_dir):
                        os.makedirs(self.checkpoint_dir)
                    self.saver.save(self.sess, self.checkpoint_dir, global_step=counter)
