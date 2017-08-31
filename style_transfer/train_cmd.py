from __future__ import print_function
import tensorflow as tf
import numpy as np
import itertools
from glob import glob

from ops import *
from utils import *
from six.moves import xrange
from vgg_network import VGG


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("style", "", "the style image path")
tf.flags.DEFINE_string("train", "", "the training images path")
tf.flags.DEFINE_string("model", "", "the path to save model and summary")
tf.flags.DEFINE_string("vgg", "", "the vgg data path")

tf.flags.DEFINE_integer("epochs", 10, "the default number of epochs")
tf.flags.DEFINE_integer("batch_size", 64, "the training batch size")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate for Adam Optimizer")
tf.flags.DEFINE_float("content_weight", 1, "the content loss weight")
tf.flags.DEFINE_float("style_weight", 5, "the style loss weight")

IMAGE_SIZE = 256
HALF_SIZE = IMAGE_SIZE / 2
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]


def dataset_files(root):
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


class FastStyleTransfer(object):
    content_layer = 'relu3_3'
    style_layers = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')

    def __init__(self, sess, vgg, style, model,
                 epochs, batch_size, learning_rate,
                 content_weight, style_weight):
        self.sess = sess
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.style_image = imread(style, IMAGE_SIZE)
        self.style_image = np.ndarray.reshape(self.style_image, (1,) + self.style_image.shape)
        self.vgg = VGG(vgg)
        self.style_loss_net = self.vgg.net(self.vgg.preprocess(self.style_image), name='style')

        self.input = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_SIZE, IMAGE_SIZE, 3], name="input")
        self.input_net = self.vgg.net(self.vgg.preprocess(self.input), name='input_net')
        self.stylized_image = self.transform_fn(self.input)
        self.stylized_net = self.vgg.net(self.vgg.preprocess(self.stylized_image), name='stylized_net')

        content_diff = self.input_net[self.content_layer] - self.stylized_net[self.content_layer]
        self.content_loss = 2 * content_weight * tf.reduce_mean(tf.multiply(content_diff, content_diff))
        self.style_loss = self.style_loss_fn(self.stylized_net, style_weight)
        self.loss = self.content_loss + self.style_loss

        self.style_image_summary = tf.summary.image("style_image", self.style_image / 255.)
        self.input_summary = tf.summary.image("input_summary", self.input)
        self.stylized_image_summary = tf.summary.image("stylized_image", self.stylized_image)
        self.cl_summary = tf.summary.scalar("content_loss", self.content_loss)
        self.sl_summary = tf.summary.scalar("style_loss", self.style_loss)
        self.loss_summary = tf.summary.scalar("loss", self.loss)

        self.summary = tf.summary.merge([self.style_image_summary, self.input_summary, self.stylized_image_summary,
                                         self.cl_summary, self.sl_summary, self.loss_summary])
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.writer = tf.summary.FileWriter(os.path.join(model, "logs"), self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=3)

    def style_loss_fn(self, transform_loss_net, weight):
        loss = 0
        for layer in self.style_layers:
            style_feature = self.style_loss_net[layer][0]
            h, w, number = int(style_feature.shape[0]), int(style_feature.shape[1]), int(style_feature.shape[2])
            style_feature = tf.reshape(tensor=style_feature, shape=[h * w, number])
            style_image_gram = tf.matmul(tf.transpose(style_feature), style_feature) / (h*w*number)

            input_feature = transform_loss_net[layer]
            h, w, number = int(input_feature.shape[1]), int(input_feature.shape[2]), int(input_feature.shape[3])
            input_feature = tf.reshape(input_feature, (self.batch_size, h * w, number))
            input_image_gram = tf.matmul(tf.transpose(input_feature, perm=[0,2,1]), input_feature) / (h*w*number)

            loss += weight * 2 * tf.reduce_mean(tf.multiply(input_image_gram - style_image_gram, input_image_gram - style_image_gram))
        return loss

    def transform_fn(self, image):
        image = image / 255.0
        conv1 = tf.nn.relu(conv2d(image, 32, 9, 9, 1, 1, name='transform_conv1'))
        conv2 = tf.nn.relu(conv2d(conv1, 64, 3, 3, 2, 2, name='transform_conv2'))
        conv3 = tf.nn.relu(conv2d(conv2, 128, 3, 3, 2, 2, name='transform_conv3'))
        resid1 = residual_block(conv3, 128, 3, 3, name='residual1')
        resid2 = residual_block(resid1, 128, 3, 3, name='residual2')
        resid3 = residual_block(resid2, 128, 3, 3, name='residual3')
        resid4 = residual_block(resid3, 128, 3, 3, name='residual4')
        resid5 = residual_block(resid4, 128, 3, 3, name='residual5')
        deconv1 = conv2d_transpose(resid5, [self.batch_size, HALF_SIZE, HALF_SIZE, 64], 3, 3, name='deconv1')
        deconv2 = conv2d_transpose(deconv1, [self.batch_size, IMAGE_SIZE, IMAGE_SIZE, 32], 3, 3, name="deconv2")
        conv4 = conv2d(deconv2, 3, 9, 9, 1, 1, name='transform_conv4')
        return tf.nn.tanh(conv4) * 127.5 + 255./2

    def train(self, train_path):
        self.sess.run(tf.global_variables_initializer())
        data = dataset_files(train_path)
        counter = 0
        for epoch in range(self.epochs):
            np.random.shuffle(data)
            for idx in range(len(data) // self.batch_size):
                counter += 1
                batch_files = data[idx * self.batch_size:(idx+1) * self.batch_size]
                batch = [imread(batch_file, IMAGE_SIZE) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                summary_str, content_loss, style_loss, _ = self.sess.run(
                    [self.summary, self.content_loss, self.style_loss, self.train_op],
                    feed_dict={self.input: batch_images})
                self.writer.add_summary(summary_str, counter)
                print(epoch, idx, "content_loss:", content_loss, "style_loss:", style_loss)


def main(argv):
    with tf.Session() as sess:
        style_transfer = FastStyleTransfer(
            sess=sess,
            vgg=FLAGS.vgg,
            style=FLAGS.style,
            model=FLAGS.model,
            epochs=FLAGS.epochs,
            batch_size=FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            content_weight=FLAGS.content_weight,
            style_weight=FLAGS.style_weight)
        style_transfer.train(FLAGS.train)

if __name__ == "__main__":
    tf.app.run()