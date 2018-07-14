# -*- coding: UTF-8 -*-
# write by feng
import tensorflow as tf
import pandas as pd
import argparse

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def maybeDownload():
    pathTrain = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)
    pathTest = tf.keras.utils.get_file(TEST_URL.split('/')[-1],TEST_URL)
    return pathTrain,pathTest
def loadData(label_name='Species'):
    pathTrain ,pathTest =maybeDownload()
    Train = pd.read_csv(pathTrain,names=CSV_COLUMN_NAMES,header=0)
    Test = pd.read_csv(pathTest,names=CSV_COLUMN_NAMES,header=0)
    Train_X,Train_Y = Train,Train.pop(label_name)
    print(Train_X)
    Test_X,Test_Y = Test, Test.pop(label_name)
    return (Train_X,Train_Y), (Test_X,Test_Y)
def trainInFunc (features ,labels ,batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batchsize)
    return dataset
def testInFunc (features, labels, batchsize):
    features = dict(features)
    if labels is None:
        input = features
    else:
        input = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(input)
    assert batchsize is not None, "batch_size must not be None"
    dataset = dataset.batch(batchsize)
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
def main(argv):
    arg = parser.parse_args(argv[1:])
    (Train_X, Train_Y) ,(Test_X, Test_Y) = loadData()
    feature_column = []
    for key in Test_X.keys():
        feature_column.append(tf.feature_column.numeric_column(key=key))
    classifier = tf.estimator.DNNClassifier(hidden_units=[10,10],feature_columns=feature_column,n_classes=3)
    classifier.train(lambda :trainInFunc(Train_X,Train_Y,100),steps=1000)
    accuracy = classifier.evaluate(lambda :testInFunc(Test_X,Test_Y,100))
    print(accuracy)
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

