import cv2
import csv
from config import *
import numpy as np
from os.path import join
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


DIR = '/Users/dwang/tensorflow_demo/'


def split_train_val(file, test_size=0.2):
    with open(file, 'r') as input:
        reader = csv.reader(input)
        data = [row for row in reader][1:]
    train, test = train_test_split(data, test_size=test_size, random_state=1)
    return train, test


def preprocess(filename):
    frame_bgr = cv2.imread(filename)
    frame_cropped = frame_bgr[CROP_HEIGHT, :, :]
    frame_resized = cv2.resize(frame_cropped, dsize=(WIDTH, HEIGHT))
    if CHANNELS == 1:
        frame_resized = np.expand_dims(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)[:,:,0], 2)
    return frame_resized.astype('float32')


def load_data_batch(data, batchsize, augment_data, bias):
    """
    :param data: list of training data
    :param batchsize:
    :param data_dir: directory of frames
    :param augment_data:
    :param bias: bias for balancing ground truth distribution.
    :return:
    """
    h, w, c = HEIGHT, WIDTH, CHANNELS
    x = np.zeros(shape=(batchsize, h, w, c), dtype=np.float32)
    y_steer = np.zeros(shape=(batchsize, ), dtype=np.float32)
    shuffled_data = shuffle(data)

    cnt = 0
    while cnt < batchsize:
        ct_path, lt_path, rt_path, steer, _, brake, speed = shuffled_data.pop()
        steer = np.float32(steer)
        camera = random.choice(['frontal', 'left', 'right'])
        if camera == 'frontal':
            frame = preprocess(join(DIR, ct_path.strip()))
            steer = steer
        elif camera == 'left':
            frame = preprocess(join(DIR, lt_path.strip()))
            steer += DATA_CORRECTION
        elif camera == 'right':
            frame = preprocess(join(DIR, rt_path.strip()))
            steer -= DATA_CORRECTION

        if augment_data:
            if random.random() < 0.5:
                frame = frame[:, ::-1, :]
                steer *= -1.
            steer += np.random.normal(loc=0, scale=AUGMENT_STEER_SIGMA)
            if CHANNELS == 3:
                frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
                frame[:, :, 2] *= random.uniform(AUGMENT_VALUE_MIN, AUGMENT_VALUE_MAX)
                frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
                frame = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)

        steer_magnitude_thresh = np.random.rand()
        if (abs(steer) + bias) < steer_magnitude_thresh:
            pass
        else:
            x[cnt] = frame
            y_steer[cnt] = steer
            cnt += 1

    return x, y_steer


def generate_data_batch(data, batchsize=BATCH_SIZE, augment_data=True, bias=0.5):
    while True:
        x, y = load_data_batch(data, batchsize, augment_data, bias)
        yield x, y


if __name__ == '__main__':
    train_data, test_data = split_train_val('/Users/dwang/self-driving-car/project_3_behavioral_cloning/data/driving_log.csv')