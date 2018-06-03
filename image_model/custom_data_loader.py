from functools import partial, update_wrapper
import os

import numpy as np
from PIL import Image
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, ImageDataGenerator


class Data(object):
    def __init__(self, path, is_attribute=False):
        self.categories = {}
        self.attibutes = {}
        self.num_categories = 0
        self.num_images = 0

        cnt = 0
        for l in open(path):
            cnt += 1
            if cnt <= 2:
                continue

            items = l.strip().split()
            items = filter(lambda item: item, items)
            self.num_images += 1

            if not is_attribute:
                category = int(items[1])
                if category >= self.num_categories:
                    self.num_categories = category + 1
                filename = items[0]
                if category not in self.categories:
                    self.categories[category] = []
                self.categories[category].append(filename)
            else:
                filename = items[0]
                self.num_categories = max(len(items), self.num_categories)
                self.attibutes[filename] = items[1:]

        print "Categories size %d, total images %d" % (self.num_categories, self.num_images)

    def categories_size(self):
        return self.num_categories


def preprocess_image(path, img_size):
    if not os.path.isfile(path):
        return np.zeros((1, img_size, img_size, 3), dtype=np.float32)
    img = Image.open(path).convert('RGB')
    img = img.resize((img_size, img_size))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


class SimpleSampler(object):
    def __init__(self,
                 data_path,
                 category_label_path,
                 attribute_label_path,
                 image_size=299,
                 batch_size=32,
                 num_per_category=32):
        self.category_data = Data(path=category_label_path)
        self.attribute_data = Data(path=attribute_label_path, is_attribute=True)
        self.category_size = self.category_data.categories_size()
        self.attribute_size = self.attribute_data.categories_size()
        self.label_size = self.category_size + self.attribute_size
        self.steps_per_epoch = num_per_category * len(self.category_data.categories) // batch_size

        self.num_per_category = num_per_category
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(horizontal_flip=True)
        self.sample()

    def sample(self):
        x = []
        y = []
        for c in self.category_data.categories:
            choice = np.random.choice(self.category_data.categories[c], self.num_per_category)
            for file in choice:
                x.append(preprocess_image(os.path.join(self.data_path, file), self.image_size)[0])
                label = np.zeros(dtype=np.float32, shape=[self.label_size])
                label[c] = 1
                attributes = self.attribute_data.attibutes[file]
                for i, v in enumerate(attributes):
                    if v == '1':
                        label[i + self.category_size] = 1
                y.append(label)
        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def get_generator(self):
        return self.datagen.flow(x=self.x,
                                 y=self.y,
                                 batch_size=self.batch_size,
                                 shuffle=True)