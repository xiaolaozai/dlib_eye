# -*- coding: UTF-8 -*-
# write by feng
import tensorflow as tf
import numpy as np
import argparse

learning_rate = 0.1
num_steps = 5000
batch_size = 10
display_step = 100
num_examples = 662

num_input = 9
num_classe = 3
n_hidden_1 = 20
n_hidden_2 = 20

def read_my_file_format(filename_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[0.0] for i in range(num_input)]
    record_defaults.append([0])
    date = tf.decode_csv(
      value, record_defaults=record_defaults)
    lable = date.pop()
    features = tf.stack(date)
    return features, lable
'''
def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_my_file_format(filename_queue)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch
'''
def read_my_file(filenames):
    record_defaults = [[0.0] for i in range(num_input)]
    record_defaults.append([0])
    date = tf.decode_csv(
      value, record_defaults=record_defaults)
    features = tf.reshape(date[:-1],shape(num_input,))
    label = tf.reshape(date[-1],shape=())
    #lable = date.pop()
    #features = tf.stack(date)
    return features, lable

filenames = ["train/train.csv"]
#example_bat, label_bat = read_my_file(filenames)
train_dateset = tf.train.TextLineReader(filenames)
train_dateset = train_dateset.map(read_my_file)
train_dateset = train_dateset.shuffle(buffer_size=1000)
train_dateset = train_dateset.bach(batch_size)

features , label =
X = tf.placeholder("float32",[None,num_input])
Y = tf.placeholder("float32",[None,num_classe])

weights = {
    'h1':tf.Variable(tf.random_normal([num_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,num_classe]))
}
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([num_classe]))
}

def neural_net(x):
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    out_layer = tf.matmul(layer_2,weights['out']) + biases['out']
    return out_layer

logits = neural_net(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

#start training 
with tf.Session() as sess:
    sess.run(init)
    for step in range(num_steps+1):
        
        sess.run(train_op,feed_dict={X: batch_x, Y: batch_y})

        if step % display_step==0 or step ==1:
                loss, acc=sess.run([loss_op,accuracy],feed_dict={X: batch_x, \
                    Y: batch_y})
                print("step "+ str(step) +", Minibatch Loss= "+ \
                    "{:.4f}".format(loss)+", Training Accuracy = "+ \
                    "{:.3f}".format(acc))

    print("Optimization Finished!")

    #print("Testing accuracy:",\
    #    sess.run(accuracy,feed_dict={X:  , Y:  }))


