from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import tensorflow as tf
from six.moves import xrange

from ops import *
from utils import *


SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]
HATS_IMGS = "hats_imgs"
COMPLETED = "completed"
LOGS = "logs"


def dataset_files(root):
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


class DCGAN(object):
    def __init__(self,
                  sess,
                  image_size=64,
                  is_crop=False,
                  batch_size=64,
                  sample_size=64,
                  lowres=8, # low resolution image / mask shrink factor.
                  z_dim=128, # dimension of dim for z.
                  gf_dim=64, # dimension of gen filters in first conv layer.
                  df_dim=64, # dimension of discrim filters in first conv layer.
                  gfc_dim=1024, # dimension of gen unit for fully connected layer.
                  dfc_dim=1024, # dimension of discrim units for fully connected layer.
                  c_dim=3, # dimension of image color.
                  checkpoint_dir=None,
                  lam=0.1):
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]

        self.lowres = lowres
        self.lowres_size = image_size // lowres
        self.lowres_shape = [self.lowres_size, self.lowres_size, c_dim]

        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.lam = lam
        self.c_dim = c_dim

        self.d_bns = [batch_norm(name='d_bn{}'.format(i,)) for i in range(4)]
        log_size = int(math.log(image_size) / math.log(2))
        self.g_bns = [batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size)]

        self.checkpoint_dir = checkpoint_dir
        self.model_name = "DCGAN.model"
        self.build_model()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.lowres_images = tf.reduce_mean(
            tf.reshape(self.images, [self.batch_size, self.lowres_size, self.lowres, self.lowres_size, self.lowres, self.c_dim]),
            [2, 4])
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        # Generate fake images.
        self.G = self.generator(self.z)

        # Evaluate true images and fake images.
        self.D, self.D_logits = self.discriminator(self.images)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        # Optimize discriminator for true images.
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        # Optimize discriminator for fake images.
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        # Optimize generator for discriminator.
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver(max_to_keep=10)

        # Image complete
        self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam * self.perceptual_loss

        # Gradient for generator vector z
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def train(self, config):
        data = dataset_files(config.dataset)
        np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()

        self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print("""Loading existing model""")
        else:
            print("""Initalizing new model""")

        for epoch in xrange(config.epoch):
            data = dataset_files(config.dataset)
            batch_idxs = min(len(data), config.train_size) // self.batch_size
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx * config.batch_size : (idx + 1) * config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.images: batch_images,
                                                          self.z: batch_z,
                                                          self.is_training: True})
                self.writer.add_summary(summary_str, counter)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z,
                                                          self.is_training:True})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z,
                                                          self.is_training: True})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
                errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})

                counter += 1
                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.G, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images, self.is_training: False}
                    )
                    save_images(samples, [8, 8],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, images, reuse=False):
        """
        Images shape: [None, 64, 64, 3]
        H0: 32 * 32 * df_dim (64)
        H1: 16 * 16 * 128
        H2: 8 * 8 * 256
        H3: 4 * 4 * 512
        H4: Linear Output = [None, 4, 4, 512] * [8192, 1]
        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            h0 = lrelu(conv2d(images, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim * 2, name='d_h1_conv'), self.is_training))
            h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim * 4, name='d_h2_conv'), self.is_training))
            h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim * 8, name='d_h3_conv'), self.is_training))
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_linear')
            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        """
        Tensor shape:  [None, 128]
        H0: [None, 128] * [128, 64 * 8 * 4 * 4], reshape to [None, 4, 4, 512]
        H1: deconv [None, 4, 4, 512] -> [None, 8, 8, 256]
        H2: deconv [None, 8, 8, 256] -> [None, 16, 16, 128]
        H3: deconv [None, 16, 16, 128] -> [None, 32, 32, 64]
        H4: deconv [None, 32, 64, 64] -> [None, 64, 64, 3]
        """
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * 4 * 4, 'g_h0_lin', with_w=True)
            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))

            i = 1
            depth_mul = 8
            size = 8
            while size < self.image_size:
                hs.append(None)
                name = 'g_h{}'.format(i)
                hs[i] = conv2d_transpose(hs[i - 1],
                                         [self.batch_size, size, size, self.gf_dim * depth_mul],
                                         name=name)
                hs[i] = tf.nn.relu(self.g_bns[i](hs[i], self.is_training))

                i += 1
                depth_mul //= 2
                size *= 2

            hs.append(None)
            name = 'g_h{}'.format(i)
            hs[i] = conv2d_transpose(hs[i - 1],
                                     [self.batch_size, size, size, 3],
                                     name=name)
            return tf.nn.tanh(hs[i])

    def complete(self, config):
        def make_dir(name):
            p = os.path.join(config.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)
        make_dir(HATS_IMGS)
        make_dir(COMPLETED)
        make_dir(LOGS)

        # Load GAN model from existing files.
        tf.initialize_all_variables().run()
        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        # Setup image mask
        assert(config.centerScale <= 0.5)
        mask = np.ones(self.image_shape)
        l = int(self.image_size*config.centerScale)
        u = int(self.image_size*(1.0-config.centerScale))
        mask[l:u, l:u, :] = 0.0

        batchSz = min(self.batch_size, len(config.imgs))
        batch = [get_image(file, self.image_size, is_crop=self.is_crop) for file in config.imgs[:batchSz]]
        batch_images = np.array(batch).astype(np.float32)
        if batchSz < self.batch_size:
            padSz = ((0, int(self.batch_size - batchSz)), (0, 0), (0, 0), (0, 0))
            batch_images = np.pad(batch_images, padSz, 'constant')
            batch_images = batch_images.astype(np.float32)

        zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        m = 0
        v = 0
        nRows = np.ceil(batchSz/8)
        nCols = min(8, batchSz)
        save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                    os.path.join(config.outDir, 'before.png'))
        # The product of `x1` and `x2`, element-wise.
        masked_images = np.multiply(batch_images, mask)
        save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
                    os.path.join(config.outDir, 'masked.png'))

        for img in range(batchSz):
            with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'a') as f:
                f.write('iter loss ' +
                        ' '.join(['z{}'.format(zi) for zi in range(self.z_dim)]) +'\n')
        for i in xrange(config.nIter):
            fd = {
                self.z: zhats,
                self.mask: mask,
                self.images: batch_images,
                self.is_training: False
            }
            run = [self.complete_loss, self.grad_complete_loss, self.G]
            loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

            for img in range(batchSz):
                with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                    f.write('{} {} '.format(i, loss[img]).encode())
                    np.savetxt(f, zhats[img:img+1])

            if i % config.outInterval == 0:
                print(i, np.mean(loss[0:batchSz]))
                imgName = os.path.join(config.outDir, 'hats_imgs/{:04d}.png'.format(i))
                save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)
                inv_masked_hat_images = np.multiply(G_imgs, 1.0-mask)
                completed = masked_images + inv_masked_hat_images
                imgName = os.path.join(config.outDir, 'completed/{:04d}.png'.format(i))
                save_images(completed[:batchSz,:,:,:], [nRows,nCols], imgName)

            # Optimize single completion with Adam
            m_prev = np.copy(m)
            v_prev = np.copy(v)
            m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
            v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
            m_hat = m / (1 - config.beta1 ** (i + 1))
            v_hat = v / (1 - config.beta2 ** (i + 1))
            zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
            zhats = np.clip(zhats, -1, 1)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False