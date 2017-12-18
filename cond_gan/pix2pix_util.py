from __future__ import division
import scipy.misc
import numpy as np
from utils import imread


def load_data(image_path, flip=True, is_test=False):
    a, b = load_image(image_path)
    a, b = preprocess(a, b, flip=flip, is_test=is_test)
    a = a / 127.5 - 1.
    b = b / 127.5 - 1.
    ab = np.concatenate((a, b), axis=2)
    return ab

def load_image(image_path):
    img = imread(image_path)
    w = int(img.shape[1])
    mid = int(w / 2)
    a = img[:, 0:mid]
    b = img[:, mid:w]
    return a, b

def preprocess(a, b, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        a = scipy.misc.imresize(a, [fine_size, fine_size])
        b = scipy.misc.imresize(b, [fine_size, fine_size])
    else:
        a = scipy.misc.imresize(a, [load_size, load_size])
        b = scipy.misc.imresize(b, [load_size, load_size])
        h = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        a, b = a[h:h+fine_size, w:w+fine_size], b[h:h+fine_size, w:w+fine_size]
        if flip and np.random.random() > 0.5:
            a, b = np.fliplr(a), np.fliplr(b)
    return a, b
