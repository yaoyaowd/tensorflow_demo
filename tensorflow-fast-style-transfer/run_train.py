import tensorflow as tf
import numpy as np
import utils
import vgg19
import style_transfer_trainer
import os

import argparse

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("style", "", "the style image path")
tf.flags.DEFINE_string("train", "", "the training images path")
tf.flags.DEFINE_string("output", "", "the path to save model and summary")
tf.flags.DEFINE_string("vgg_model", "", "the vgg data path")

tf.flags.DEFINE_integer("num_epochs", 10, "the default number of epochs")
tf.flags.DEFINE_integer("batch_size", 16, "the training batch size")
tf.flags.DEFINE_float("learn_rate", 1e-3, "learning rate for Adam Optimizer")
tf.flags.DEFINE_float("content_weight", 7.5, "the content loss weight")
tf.flags.DEFINE_float("style_weight", 100, "the style loss weight")
tf.flags.DEFINE_float("tv_weight", 200, "the total variance loss weight")
tf.flags.DEFINE_integer("checkpoint_every", 100, "save a trained model every this number of iterations")
tf.flags.DEFINE_integer("max_size", 0, "the maximum width or height of input images")

CONTENT_LAYERS_NAME = ['relu4_2']
STYLE_LAYERS_NAME = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
CONTENT_LAYER_WEIGHTS = [1.0]
STYLE_LAYER_WEIGHTS = [.2,.2,.2,.2,.2]

"""add one dim for batch"""
# VGG19 requires input dimension to be (batch, height, width, channel)
def add_one_dim(image):
    shape = (1,) + image.shape
    return np.reshape(image, shape)

"""main"""
def main(argv):
    model_file_path = os.path.join(FLAGS.vgg_model, vgg19.MODEL_FILE_NAME)
    vgg_net = vgg19.VGG19(model_file_path)
    content_images = utils.get_files(FLAGS.train)
    style_image = utils.load_image(FLAGS.style)

    # create a map for content layers info
    CONTENT_LAYERS = {}
    for layer, weight in zip(CONTENT_LAYERS_NAME, CONTENT_LAYER_WEIGHTS):
        CONTENT_LAYERS[layer] = weight

    # create a map for style layers info
    STYLE_LAYERS = {}
    for layer, weight in zip(STYLE_LAYERS_NAME, STYLE_LAYER_WEIGHTS):
        STYLE_LAYERS[layer] = weight

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        trainer = style_transfer_trainer.StyleTransferTrainer(
            session=sess,
            content_layer_ids=CONTENT_LAYERS,
            style_layer_ids=STYLE_LAYERS,
            content_images=content_images,
            style_image=add_one_dim(style_image),
            net=vgg_net,
            num_epochs=FLAGS.num_epochs,
            batch_size=FLAGS.batch_size,
            content_weight=FLAGS.content_weight,
            style_weight=FLAGS.style_weight,
            tv_weight=FLAGS.tv_weight,
            learn_rate=FLAGS.learn_rate,
            save_path=FLAGS.output,
            check_period=FLAGS.checkpoint_every,
            max_size=FLAGS.max_size or None)

        trainer.train()

if __name__ == "__main__":
    tf.app.run()
