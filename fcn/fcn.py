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
tf.flags.DEFINE_string("vgg_model", "", "Path to vgg model mat")

VGG_MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
MAX_ITERATION = 100000
NUM_OF_CLASSES = 151
IMAGE_SIZE = 224

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    )
    net = {}
    for i, name in enumerate(layers):
        if name.startswith('conv'):
            kernels, bias = weights[i][0][0][0][0]
            kernels = restore_variable(np.transpose(kernels, (1,0,2,3)), name=name + '_w')
            bias = restore_variable(bias.reshape(-1), name=name + '_b')
            image = tf.nn.bias_add(tf.nn.conv2d(image, kernels, strides=[1,1,1,1], padding="SAME"), bias)
        elif name.startswith('relu'):
            image = tf.nn.relu(image, name=name)
        elif name.startswith('pool'):
            image = avg_pool(image, 2, 2, 2, 2)
        net[name] = image
        print(name, image)
    return net


def train(learning_rate, loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)


def inference(model_dir, image, keep_prob):
    model_data = get_vgg_model(model_dir, VGG_MODEL_URL)
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0,1))
    weights = np.squeeze(model_data['layers'])
    processed_image = image + mean_pixel

    with tf.variable_scope("inference"):
        image_net= vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
        pool5 = max_pool(conv_final_layer, 2, 2)

        conv6 = conv2d(pool5, 1024, 7, 7, 1, 1, name='conv6')
        relu6 = tf.nn.relu(conv6, name='relu6')
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        conv7 = conv2d(relu_dropout6, 1024, 1, 1, 1, 1, name='conv7')
        relu7 = tf.nn.relu(conv7, name='relu7')
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        conv8 = conv2d(relu_dropout7, NUM_OF_CLASSES, 1, 1, 1, 1, name='conv8')

        deconv1 = conv2d_transpose(conv8, image_net["pool4"].shape, 4, 4, name="deconv1")
        fuse_1 = tf.add(deconv1, image_net["pool4"], name="fuse_1")

        deconv2 = conv2d_transpose(fuse_1, image_net["pool3"].shape, 4, 4, name="deconv2")
        fuse_2 = tf.add(deconv2, image_net["pool3"], name="fuse_2")

        deconv_shape = [int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), NUM_OF_CLASSES]
        deconv3 = conv2d_transpose(fuse_2, deconv_shape, 16, 16, 8, 8, name="deconv3")

        annotation_pred = tf.argmax(deconv3, dimension=3, name="prediction")
    return tf.expand_dims(annotation_pred, dim=3), deconv3

def main(argv):
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    image = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(FLAGS.vgg_model, image, keep_prob=keep_prob)
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
        train_loss, _, summary_str = sess.run([loss, train_op, summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, itr)
        print("Time: %d, Step: %d, Train loss: %g" % (time.time(), itr, train_loss))
        if itr % 10 == 0 and itr > 0:
            saver.save(sess, FLAGS.model_dir + "model.ckpt", itr)

if __name__ == "__main__":
    tf.app.run()