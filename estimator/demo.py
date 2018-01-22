import os
import six.moves.urllib.request as request
import tensorflow as tf

print tf.__version__

PATH = "/tmp/"
MODEL_PATH = "/tmp/model/"
FILE_TRAIN = PATH + "iris_training.csv"
FILE_TEST = PATH + "iris_test.csv"
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"


def downloadDataset(url, file):
    if os.path.exists(file):
        return
    data = request.urlopen(url).read()
    with open(file, 'wb') as f:
        f.write(data)
downloadDataset(URL_TRAIN, FILE_TRAIN)
downloadDataset(URL_TEST, FILE_TEST)

tf.logging.set_verbosity(tf.logging.INFO)


feature_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']

def input_fn(file_path, shuffle=False, repeat_count=1):
    def decode_csv(line):
        ret = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = ret[-1]
        return dict(zip(feature_names, ret[:-1])), label

    dataset = (tf.data.TextLineDataset(file_path).skip(1).map(decode_csv))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10,10],
    n_classes=3,
    model_dir=MODEL_PATH)
classifier.train(input_fn=lambda: input_fn(FILE_TRAIN, True, 8))


evaluate_result = classifier.evaluate(input_fn=lambda: input_fn(FILE_TEST, False, 1))
for key in evaluate_result:
    print key, evaluate_result[key]


predict_results = classifier.predict(input_fn=lambda: input_fn(FILE_TEST, False, 1))
for prediction in predict_results:
    print(prediction["class_ids"][0])
