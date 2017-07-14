import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/Users/dwang/Downloads/mnist/", one_hot=True)
print "training data", mnist.train.num_examples
print "validating data", mnist.validation.num_examples
print "testing data", mnist.test.num_examples


batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print xs.shape
print ys.shape