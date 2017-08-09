from __future__ import division

import os
import time
import itertools
import numpy as np
import tensorflow as tf

from glob import glob

from ops import lrelu, conv2d, conv2d_transpose, linear
from utils import get_image


D_ITERATIONS = 1
G_ITERATIONS = 2
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

def dataset_files(root):
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


class WGANVector(object):
    def __init__(self,
                 sess,
                 learning_rate=0.0001,
                 beta1=0.5,
                 batch_size=32,
                 image_size=256,
                 lam=0.4,
                 checkpoint_dir=None):
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_shape = [image_size, image_size, 3]
        self.lam = lam
        self.checkpoint_dir = checkpoint_dir
        self.global_step = tf.Variable(0, trainable=False)

        self.images = tf.placeholder(tf.float32, [batch_size] + self.image_shape, name='images')
        self.images_summary = tf.summary.image("image", self.images)
        self.masks = tf.placeholder(tf.float32, [batch_size] + self.image_shape, name='masks')
        self.masked_images = tf.multiply(self.images, self.masks)
        self.masked_images_summary = tf.summary.image("masked_image", self.masked_images)
        self.G = self.generator(self.masked_images)
        self.G_summary = tf.summary.image("g", self.G)
        self.epsilon = tf.random_uniform([], 0.0, 1.0)
        self.G_epsilon = self.epsilon * self.images + (1 - self.epsilon) * self.G

        self.D_real = self.discriminator(self.images, self.image_size)
        self.D_fake = self.discriminator(self.G, self.image_size, reuse=True)
        self.D_epsilon = self.discriminator(self.G_epsilon, self.image_size, reuse=True)
        self.gradients = tf.gradients(self.D_epsilon, self.G_epsilon)[0]
        self.gradient_penalty = 10.0 * tf.square(tf.norm(self.gradients, ord=2) - 1.0)

        self.d_loss_real = tf.reduce_mean(self.D_real)
        self.d_loss_fake = tf.reduce_mean(self.D_fake)
        self.d_loss = self.d_loss_fake - self.d_loss_real + self.gradient_penalty
        self.d_loss_real_summary = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_summary = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.gradient_penalty_summary = tf.summary.scalar("d_loss_gp", self.gradient_penalty)
        self.d_loss_summary = tf.summary.scalar("d_loss", self.d_loss)
        self.d_summary = tf.summary.merge([
            self.images_summary,
            self.d_loss_real_summary, self.d_loss_fake_summary, self.gradient_penalty_summary,
            self.d_loss_summary])

        #self.g_loss_mse = tf.reduce_mean(tf.contrib.layers.flatten(tf.square(self.G - self.images)))
        #self.g_loss_mse_summary = tf.summary.scalar("g_mse", self.g_loss_mse)
        #self.g_loss = -(1 - self.lam) * self.d_loss_fake + self.lam * self.g_loss_mse
        self.g_loss = -self.d_loss_fake
        self.g_loss_summary = tf.summary.scalar("g", self.g_loss)
        #self.g_summary = tf.summary.merge([
        #    self.G_summary, self.masked_images_summary, self.g_loss_mse_summary, self.g_loss_summary])
        self.g_summary = tf.summary.merge([
            self.G_summary, self.masked_images_summary, self.g_loss_summary])

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
            self.d_loss, var_list=self.d_vars, global_step=self.global_step)
        self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
            self.g_loss, var_list=self.g_vars)

        self.writer = tf.summary.FileWriter(os.path.join(self.checkpoint_dir, "logs"), self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=10)

    def train(self, config):
        tf.global_variables_initializer().run()
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print("""Loading existing model""")
        else:
            print("""Initalizing new model""")

        for epoch in range(config.epoch):
            data = dataset_files(config.dataset)
            np.random.shuffle(data)
            batch_idx = min(len(data), config.train_size) // self.batch_size
            for idx in range(batch_idx):
                batch_files = data[idx * config.batch_size: (idx + 1) * config.batch_size]
                batch = [get_image(batch_file, self.image_size) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                masks = np.ones([config.batch_size, config.image_size, config.image_size, 3])
                masks[:, 16:48, 16:48, :] = 0.0

                for i in range(D_ITERATIONS):
                    _, d_loss, summary_str = self.sess.run(
                        [self.d_optim, self.d_loss, self.d_summary],
                        feed_dict={self.images: batch_images, self.masks: masks})
                    self.writer.add_summary(summary_str, self.global_step.eval())

                for i in range(G_ITERATIONS):
                    _, g_loss, summary_str = self.sess.run(
                        [self.g_optim, self.g_loss, self.g_summary],
                        feed_dict={self.images: batch_images, self.masks: masks})
                    self.writer.add_summary(summary_str, self.global_step.eval())
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                        epoch, idx, batch_idx, time.time() - start_time, d_loss, g_loss))

                if idx % 10 == 0:
                    self.save(config.checkpoint_dir, self.global_step.eval())

    def generator(self, images):
        with tf.variable_scope("generator"):
            g_h0 = tf.nn.relu(conv2d(images, 16, name='g_encode_0'))
            g_h1 = tf.nn.relu(conv2d(g_h0, 32, name='g_encode_1'))
            g_h2 = tf.nn.relu(conv2d(g_h1, 64, name='g_encode_2'))
            g_flat = tf.reshape(g_h2, [self.batch_size, -1])
            g_encode = linear(g_flat, 128, 'g_encode')
            g_decode = linear(g_encode, 512 * 4 * 4, 'g_h0')
            g_h3 = tf.nn.relu(tf.reshape(g_decode, [self.batch_size, 4, 4, 512]))
            g_h4 = tf.nn.relu(conv2d_transpose(g_h3, [self.batch_size, 8, 8, 256], name='g_h1'))
            g_h5 = tf.nn.relu(conv2d_transpose(g_h4, [self.batch_size, 16, 16, 128], name='g_h2'))
            g_h6 = tf.nn.relu(conv2d_transpose(g_h5, [self.batch_size, 32, 32, 64], name='g_h3'))
            g_h7 = conv2d_transpose(g_h6, [self.batch_size, 64, 64, 3], name='g_h4')
            return tf.nn.tanh(g_h7)

    def discriminator(self, images, image_size, reuse=False):
        image_size /= 64
        with tf.variable_scope('discriminator', reuse=reuse):
            gd_h0 = lrelu(conv2d(images, 64, name="d_gd_h0_conv"))
            gd_h1 = lrelu(conv2d(gd_h0, 128, name='d_gd_h1_conv'))
            gd_h2 = lrelu(conv2d(gd_h1, 256, name='d_gd_h2_conv'))
            gd_h3 = lrelu(conv2d(gd_h2, 512, name='d_gd_h3_conv'))
            gd_h4 = lrelu(conv2d(gd_h3, 512, name='d_gd_h4_conv'))
            gd_h5 = lrelu(conv2d(gd_h4, 512, name='d_gd_h5_conv'))
            gd_h = linear(tf.reshape(
                gd_h5, [self.batch_size, int(512 * image_size * image_size)]), 64 * image_size * image_size, 'd_gd_linear')
            return linear(gd_h, 1, 'd_linear')

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, "wgan_vector"),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [32]")
flags.DEFINE_integer("image_size", 64, "The size of image to use")
flags.DEFINE_string("dataset", "lfw-aligned-64", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = WGANVector(
        sess,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        image_size=FLAGS.image_size,
        batch_size=FLAGS.batch_size,
        checkpoint_dir=FLAGS.checkpoint_dir)
    dcgan.train(FLAGS)