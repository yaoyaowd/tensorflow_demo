from __future__ import print_function
import tensorflow as tf
import data
import time

from reader import BatchReader
from ops import *
from utils import *
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 64, "batch size for training")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("data_dir", "", "path to dataset")
tf.flags.DEFINE_string("model_dir", "", "Path to vgg model mat")

MAX_ITERATION = 100000
NUM_OF_CLASSES = 151
IMAGE_SIZE = 224
PYRAMID_SIZES = [1, 2, 3, 5]

def train(learning_rate, loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

def inference(image):
    print(image)
    with tf.variable_scope("inference"):
        batch_size = int(image.shape[0])
        conv1 = conv2d(image, 64, 3, 3, 1, 1, name='conv1')
        print(conv1)
        conv2 = conv2d(conv1, 128, 3, 3, 2, 2, name='conv2')
        print(conv2)
        conv3 = conv2d(conv2, 256, 3, 3, 2, 2, name='conv3')
        print(conv3)
        conv4 = conv2d(conv3, 512, 3, 3, 2, 2, name='conv4')
        print(conv4)

        ppool = []
        for filter_size in PYRAMID_SIZES:
            with tf.name_scope("conv-%s" % filter_size):
                conv = conv2d(conv4, 512, filter_size, filter_size, 1, 1, name="p-conv-%d" % filter_size)
                print(conv)
                deconv1 = conv2d_transpose(conv, conv3.shape, filter_size, filter_size, 2, 2,
                                           name="p-deconv1-%d" % filter_size)
                print(deconv1)
                deconv2 = conv2d_transpose(deconv1, conv2.shape, filter_size, filter_size, 2, 2,
                                           name="p-deconv2-%d" % filter_size)
                print(deconv2)
                deconv3 = conv2d_transpose(deconv2, conv1.shape, filter_size, filter_size, 2, 2,
                                           name="p-deconv3-%d" % filter_size)
                print(deconv3)
                ppool.append(deconv3)

        pool_final = tf.concat(ppool, 3)
        print(pool_final)

        final_shape = [batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CLASSES]
        final_deconv = conv2d_transpose(pool_final, final_shape, 16, 16, 1, 1,
                                        name='final_deconv')
        print(final_deconv)
        annotation_pred = tf.argmax(final_deconv, dimension=3, name="prediction")
        print(annotation_pred)
    return tf.expand_dims(annotation_pred, dim=3), final_deconv

def main(argv):
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    image = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image)
    tf.summary.image("input_image", image)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8))
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8))

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.squeeze(annotation,squeeze_dims=[3])))
    tf.summary.scalar("loss", loss)

    trainable_var = tf.trainable_variables()
    train_op = train(FLAGS.learning_rate, loss, trainable_var)
    summary_op = tf.summary.merge_all()

    train_records, valid_records = data.read_dataset(FLAGS.data_dir)
    print("Train records:", len(train_records))
    print("Valid records:", len(valid_records))
    train_reader = BatchReader(train_records, {'resize': True, 'resize_size': IMAGE_SIZE})

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=10)
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, "logs"), sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored from", ckpt.model_checkpoint_path)
    else:
        print("initialize new model")

    for itr in xrange(MAX_ITERATION):
        train_images, train_annotations = train_reader.next_batch(FLAGS.batch_size)
        feed_dict = {image: train_images, annotation: train_annotations, keep_prob: 0.8}
        train_loss, _, pred_result, summary_str = sess.run([loss, train_op, pred_annotation, summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, itr)
        print("Time: %d, Step: %d, Train loss: %g" % (time.time(), itr, train_loss))
        if itr % 10 == 0 and itr > 0:
            saver.save(sess, FLAGS.model_dir + "model.ckpt", itr)
            print(pred_result[0])

if __name__ == "__main__":
    tf.app.run()