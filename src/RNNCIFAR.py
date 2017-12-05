import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
from Cifar import *
from  params import *
import time
"""
CiFar data unpacking Code Adapted from ilkarman/DeepLearningFrameworks tutorials
https://github.com/ilkarman/DeepLearningFrameworks
"""

hm_epochs = [2, 5, 10, 15, 25, 30]
error = []
time_taken = []
n_classes = 10
batch_size = 64

chunk_size = 32
n_chunks = 32
rnn_size = 128
channel_size = 3
x = tf.placeholder(tf.float16, [None, n_chunks, chunk_size, channel_size ])
y = tf.placeholder(tf.int16)

def recurrent_neural_network_model(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes], dtype=tf.float16)),
             'biases': tf.Variable(tf.random_normal([n_classes], dtype=tf.float16))}

    x = tf.transpose(x, [1, 0, 3, 2])
    x = tf.reshape(x, [-1, chunk_size*3])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True, reuse=tf.AUTO_REUSE)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float16)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.cast(prediction,tf.float32), labels=tf.cast(y, tf.int32)))
    # cost = tf.Print(cost, [cost], message="This is cost: ")
    optimizer = tf.train.AdamOptimizer(epsilon=.01, learning_rate=0.01, beta1=.8).minimize(tf.cast(cost, tf.float16))
    # optimizer = tf.Print(optimizer, [optimizer], message="This is optimizer: ")

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for k, v in enumerate(hm_epochs):
            start_time = time.time()
            for epoch in range(v):
                epoch_loss = 0
                for data, label in yield_mb(x_train, y_train, batch_size, shuffle=True):
                    _, c = sess.run([optimizer, cost],feed_dict={x: data, y: label})
                    epoch_loss += c

                # print('Epoch', v, 'completed out of',hm_epochs,'loss:',epoch_loss)
            correct = tf.nn.in_top_k(tf.cast(prediction, tf.float32), tf.cast(y, tf.int32), 1)

            accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
            error.append(accuracy.eval({x: x_test, y: y_test}))
            print("done")
            time_taken.append(time.time() - start_time)
            print('Epochs::', v, '::Accuracy:', error[k])


x_train, x_test, y_train, y_test = cifar_for_library(channel_first=False)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)
train_neural_network(x)
print(error)
print(time_taken)
