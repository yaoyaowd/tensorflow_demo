import random
from six.moves import cPickle as pickle
import glob

from utils import *

DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
PICKLE_FILENAME = "MITSceneParsing.pickle"


def read_dataset(data_dir):
    pickle_filepath = os.path.join(data_dir, PICKLE_FILENAME)
    if not os.path.exists(pickle_filepath):
        maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        result = create_image_list(os.path.join(data_dir, folder))
        print("Pickling...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Found pickle file")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training = result['training']
        validation = result['validation']
    return training, validation


def create_image_list(image_dir):
    if not os.path.exists(image_dir):
        print("Image directory not found")
        return None
    directories = ['training', 'validation']
    image_list = {}
    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.jpg')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print("No file found")
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
        random.shuffle(image_list[directory])
        print ('No. of %s files: %d' % (directory, len(image_list[directory])))
    return image_list


if __name__ == '__main__':
    read_dataset('/home/dwang/voc2012/')