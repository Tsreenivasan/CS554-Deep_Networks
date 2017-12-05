import numpy as np
import os
import sys
import tensorflow as tf
from Cifar import *
from  params import *

"""
CiFar data unpacking Code Adapted from ilkarman/DeepLearningFrameworks tutorials
https://github.com/ilkarman/DeepLearningFrameworks
"""

def create_symbol():
    conv1 = tf.layers.conv2d(tf.cast(X,tf.float32), filters=50, kernel_size=(3, 3), padding='same')
    relu1 = tf.nn.relu(conv1)
    conv2 = tf.layers.conv2d(relu1, filters=50, kernel_size=(3, 3), padding='same')
    relu2 = tf.nn.relu(conv2)
    pool1 = tf.layers.max_pooling2d(relu2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    drop1 = tf.layers.dropout(pool1, 0.25)

    conv3 = tf.layers.conv2d(drop1, filters=100, kernel_size=(3, 3), padding='same')
    relu3 = tf.nn.relu(conv3)
    conv4 = tf.layers.conv2d(relu3, filters=100, kernel_size=(3, 3), padding='same')
    relu4 = tf.nn.relu(conv4)
    pool2 = tf.layers.max_pooling2d(relu4, pool_size=(2, 2), strides=(2, 2), padding='valid')
    drop2 = tf.layers.dropout(pool2, 0.25)

    flatten = tf.reshape(drop2, shape=[-1, 100 * 8 * 8])
    fc1 = tf.layers.dense(flatten, 512, activation=tf.nn.relu)
    drop4 = tf.layers.dropout(fc1, 0.5)
    logits = tf.layers.dense(drop4, N_CLASSES, name='output')
    return tf.cast(logits, tf.float64)

def init_model(m):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.cast(m,tf.float32), labels=tf.cast(y, tf.int32))
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.MomentumOptimizer(learning_rate=LR, momentum=MOMENTUM)
    training_op = optimizer.minimize(loss)
    return training_op


# Data into format for library
#x_train, x_test, y_train, y_test = mnist_for_library(channel_first=False)
x_train, x_test, y_train, y_test = cifar_for_library(channel_first=False)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)


# Place-holders
X = tf.placeholder(tf.float64, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.int64, shape=[None])
# Initialise model
sym = create_symbol()

model = init_model(sym)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# Accuracy logging
correct = tf.nn.in_top_k(tf.cast(sym, tf.float32), tf.cast(y, tf.int32), 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float64))


import time
time_taken=[]
error = []
start_time = time.time()
hm_epochs = [2, 5, 10, 15, 25, 30]
for k, v in enumerate(hm_epochs):
    start_time = time.time()
    for epoch in range(v):
        for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):
            sess.run(model, feed_dict={X: data, y: label})
        # Log
        # acc_train = sess.run(accuracy, feed_dict={X: data, y: label})
        # print(epoch, "Train accuracy:", acc_train,time.time() - start_time)

    n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE
    y_guess = np.zeros(n_samples, dtype=np.int)
    y_truth = y_test[:n_samples]
    c = 0
    for data, label in yield_mb(x_test, y_test, BATCHSIZE):
        pred = tf.argmax(sym,1)
        output = sess.run(pred, feed_dict={X: data})
        y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = output
        c += 1
    time_taken.append(time.time() - start_time)
    error.append(sum(y_guess == y_truth)/len(y_guess))
    print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))
print(time.time()-start_time)
print(error)
print(time_taken)