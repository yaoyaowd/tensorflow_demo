from custom_data_loader import *
from functools import partial, update_wrapper

import keras
from keras import backend as K
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Activation
from keras.models import Model


def wrapped_partial(func, *args, **kwargs):
    """
    Function to freeze some portion of a function's arguments
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def loss_function(category_size, attribute_size, y_true, y_pred):
    category_loss = K.categorical_crossentropy(y_true[:,:category_size], y_pred[:,:category_size])
    attribute_loss = K.binary_crossentropy(y_true[:, category_size:], y_pred[:, category_size:])
    return category_loss + K.mean(attribute_loss, -1)

def evaluate_cate(category_size, attribute_size, y_true, y_pred):
    return categorical_accuracy(y_true[:, :category_size], y_pred[:, :category_size])

def evaluate_attr(category_size, attribute_size, y_true, y_pred):
    return top_k_categorical_accuracy(y_true[:, category_size:], y_pred[:, category_size:])


class ImageModel():

    def __init__(self, embedding_size=128, img_size=299):
        self.embedding_size=embedding_size
        self.img_size=img_size

    def get_model(self, category_size, attribute_size, learning_rate):
        inception_model = InceptionV3(include_top=False,
                                      weights="imagenet",
                                      pooling="avg")
        x = inception_model.output
        x = Dense(self.embedding_size, activation='elu', name='fc1')(x)
        x1 = Dense(category_size, name='fc_category')(x)
        x1 = Activation('softmax')(x1)
        x2 = Dense(attribute_size, name='fc_attribute')(x)
        x2 = Activation('sigmoid')(x2)
        y = keras.layers.concatenate([x1, x2], axis=-1)

        model = Model(inputs=inception_model.inputs, outputs=y)
        loss_func = wrapped_partial(loss_function, category_size, attribute_size)
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),
                      loss=loss_func,
                      metrics=[wrapped_partial(evaluate_cate, category_size, attribute_size),
                               wrapped_partial(evaluate_attr, category_size, attribute_size)])
        return model

    def train(self,
              image_path,
              log_dir,
              checkpoint_dir,
              num_epochs=100,
              batch_size=32,
              learning_rate=0.0001):
        sampler = SimpleSampler(data_path=image_path,
                                category_label_path=os.path.join(image_path, "Anno/list_category_img.txt"),
                                attribute_label_path=os.path.join(image_path, "Anno/list_attr_img.txt"),
                                batch_size=batch_size)

        self.model = self.get_model(category_size=sampler.category_size,
                                    attribute_size=sampler.attribute_size,
                                    learning_rate=learning_rate)

        tensorboard = keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)
        for i in xrange(num_epochs):
            sampler.sample()
            self.model.fit_generator(sampler.get_generator(),
                                     steps_per_epoch=sampler.steps_per_epoch,
                                     verbose=2,
                                     callbacks=[tensorboard,])
            if i > 0 and i % 10 == 0:
                self.model.save(os.path.join(checkpoint_dir, "image_model.%d.keras" % i))
