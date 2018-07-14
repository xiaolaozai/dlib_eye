# -*- coding: UTF-8 -*-
# write by feng
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

train_open_txt = open('./train/train_open.txt','rb')
train_close_txt = open('./train/train_close.txt','rb')

train = []
labels = []

print('Reading train_open.txt.....')

for txt_str in train_open_txt.readlines():
    temp = []
    # print(txt_str)
    datas = txt_str.strip()
    datas = datas.replace('[', '')
    datas = datas.replace(']', '')
    datas = datas.split(',')
    print(datas)
    for data in datas:
        #print(data)
        data = float(data)
        temp.append(data)
    #print(temp)
    train.append(temp)
    labels.append(0)

print('Reading train_close.txt...')
for txt_str in train_close_txt.readlines():
    temp = []
    datas = txt_str.strip()
    datas = datas.replace('[', '')
    datas = datas.replace(']', '')
    datas = datas.split(',')
    print(datas)
    for data in datas:
        #print(data)
        data = float(data)
        temp.append(data)
    # print(temp)
    train.append(temp)
    labels.append(1)

#for i in range(len(labels)):
#    print("{0} --> {1}".format(train[i], labels[i]))

train_close_txt.close()
train_open_txt.close()

clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
clf.fit(train, labels)
joblib.dump(clf, "./train/ear_svm.m")
print('train success!')
print('start predicing....')

print('predicting [[0.34, 0.34, 0.31],...]')
res = clf.predict([[0.34, 0.34, 0.31,0.31,0.32,0.30,0.31,0.32,0.29]])
print(res)

print('predicting [[0.19, 0.18, 0.18,0.1,0.12,0.14,0.13,0.15,0.16]]')
res = clf.predict([[0.19, 0.18, 0.18,0.10,0.12,0.14,0.13,0.15,0.16]])
print(res)
