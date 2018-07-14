# -*- coding: UTF-8 -*-
# write by feng
import tensorflow as tf
import pandas as pd
import argparse

input_num = 9
model_dir = "models/fatigue"
n_classes = 3

CSV_COLUMN_NAMES = ['eye1','eye2','eye3','head1','head2','head3',
                    'mouth1','mouth2','mouth3','Label']
filename = "./train/train.csv"

def loadData(filename):
    Train = pd.read_csv(filename,names=CSV_COLUMN_NAMES,header=0)
    Train_X,Train_Y = Train,Train.pop('Label')
    print(Train_X)
    return (Train_X,Train_Y)
def trainInFunc (features ,labels ,batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batchsize)
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', default=100, type=int, help='batch size')
parser.add_argument('-t','--train_steps', default=1000, type=int,
                    help='number of training steps')
args = vars(parser.parse_args())

def main(argv):
    (Train_X, Train_Y) = loadData(filename)
    feature_column = []
    for key in Train_X.keys():
        feature_column.append(tf.feature_column.numeric_column(key=key))
    classifier = tf.estimator.DNNClassifier(hidden_units=[10,10],\
        feature_columns=feature_column,n_classes=n_classes,model_dir=model_dir)
    classifier.train(lambda :trainInFunc(Train_X,Train_Y,args['batch_size']),\
                    steps=args['train_steps'])
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

