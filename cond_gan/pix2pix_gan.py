import os
from cond_gan.pix2pix_model import pix2pix
import tensorflow as tf

EPOCH = 200

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('dataset_name', '../dataset/edges2shoes/train/*.jpg', 'the dataset name')
tf.flags.DEFINE_string('checkpoint_dir', '../model/pix2pix/', 'the checkpoint directory name')


def main(argv):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    with tf.Session() as sess:
        model = pix2pix(sess,
                        dataset_name=FLAGS.dataset_name,
                        checkpoint_dir=FLAGS.checkpoint_dir)
        model.train()


if __name__ == '__main__':
    tf.app.run()
