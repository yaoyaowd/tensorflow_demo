# Importing data

import numpy as np
import os
import pickle

WORK_DIR = '/Users/dwang/Downloads/lab 2 data/'

def load_data():
    with open(os.path.join(WORK_DIR, 'train.p'), mode='rb') as f:
        train = pickle.load(f)
    with open(os.path.join(WORK_DIR, 'test.p'), mode='rb') as f:
        test = pickle.load(f)
    return train, test

train, test = load_data()
x_train, y_train = train['features'], train['labels']
x_test, y_test = test['features'], test['labels']
n_classes = np.unique(y_train).shape[0]

print("Training data shape =", x_train.shape)
print("Testing data shape =", x_test.shape)
print("Unique label =", n_classes)


# Preprocess data and generator

import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

def preprocess_features(x, equalize_hist=True):
    x = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:,:,0], 2) for rgb_img in x])
    if equalize_hist:
        x = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in x])
    x = np.float32(x)
    x -= np.mean(x, axis=0)
    x /= (np.std(x, axis=0) + np.finfo('float32').eps)
    return x

x_train = preprocess_features(x_train)
x_test = preprocess_features(x_test)
y_train = to_categorical(y_train, num_classes=n_classes)
y_test = to_categorical(y_test, num_classes=n_classes)

image_datagen = ImageDataGenerator(rotation_range=15.,
                                   zoom_range=.2,
                                   width_shift_range=.1,
                                   height_shift_range=.1).flow(x_train, y_train)


# Simple machine learning model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy

class Network():

    def __init__(self,
                 input_shape,
                 n_classes,
                 learning_rate=0.001):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.learning_rate = learning_rate

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=[categorical_accuracy])
        return model

    def train(self,
              datagen,
              validation_data,
              num_epoch=100,
              steps_per_epoch=40):
        self.model = self.get_model()
        self.model.fit_generator(
            generator=datagen,
            validation_data=validation_data,
            validation_steps=5,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epoch,
            verbose=2)

model = Network(x_train.shape[1:], n_classes)
model.train(image_datagen, validation_data=(x_test, y_test))
