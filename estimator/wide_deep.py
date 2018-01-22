from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import sys
import tensorflow as tf

CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket'
]

CSV_COLUMNS_DEFAULT = [
    [0], [''], [0], [''], [0],
    [''], [''], [''], [''], [''],
    [0], [0], [0], [''], ['']
]

NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

tf.app.flags.DEFINE_string("model_dir", "/tmp/census_model", "")
tf.app.flags.DEFINE_string("model_type", "wide_deep", "")
tf.app.flags.DEFINE_integer("train_epochs", 40, "")
tf.app.flags.DEFINE_integer("epochs_per_eval", 2, "")
tf.app.flags.DEFINE_integer("batch",32, "")
tf.app.flags.DEFINE_string("train_data", "/tmp/census_data/adult.data", "")
tf.app.flags.DEFINE_string("test_data", "/tmp/census_data/adult.test", "")
FLAGS = tf.app.flags.FLAGS


def input_fn(data_file, num_epochs=2, shuffle=False, batch_size=32):
    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=CSV_COLUMNS_DEFAULT)
        features = dict(zip(CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=NUM_EXAMPLES['train'])
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def build_model_columns():
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education',
        ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
         'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
         '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])
    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000)
    age_bucket = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    base_columns = [education, marital_status, relationship, workclass, occupation, age_bucket]
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_bucket, 'education', 'occupation'], hash_bucket_size=1000),]

    wide_columns = base_columns + crossed_columns
    deep_columns = [age, education_num, capital_gain, capital_loss, hours_per_week,
                    tf.feature_column.indicator_column(workclass),
                    tf.feature_column.indicator_column(education),
                    tf.feature_column.indicator_column(marital_status),
                    tf.feature_column.indicator_column(relationship),
                    tf.feature_column.embedding_column(occupation, dimension=8)]
    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]
    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=hidden_units)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_units)


def main(argv):
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(FLAGS.train_data, shuffle=True))
        result = model.evaluate(input_fn=lambda: input_fn(FLAGS.test_data, 1))
        print("Result at epoch", (n + 1) * FLAGS.epochs_per_eval)
        print("-" * 60)
        for key in sorted(result):
            print("%s: %s" % (key, result[key]))


if __name__ == '__main__':
    tf.app.run()
