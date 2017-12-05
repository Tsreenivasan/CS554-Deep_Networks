import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import time
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hm_epochs = [2,5,10,15,25,30]
error = []
time_taken = []
n_classes = 10
batch_size = 512

chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder(tf.float16, [None, n_chunks, chunk_size])
y = tf.placeholder(tf.float16)


def recurrent_neural_network_model(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes], dtype=tf.float16)),
             'biases': tf.Variable(tf.random_normal([n_classes], dtype=tf.float16))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True, reuse=tf.AUTO_REUSE)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float16)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def entropy(logits, labels):
    epsilon = tf.constant(value=0.00001, dtype=tf.float32)
    logits = tf.cast(logits, tf.float32) + epsilon
    softmax = tf.nn.softmax(logits)
    cross_entropy = -tf.reduce_sum(tf.cast(labels, tf.float32) * tf.log(softmax + epsilon), reduction_indices=[1])
    # cross_entropy = tf.Print(cross_entropy, [cross_entropy], message="This is cross_entropy: ")
    return tf.cast(cross_entropy,tf.float16)


def train_neural_network(x):
    prediction = recurrent_neural_network_model(x)
    cost = tf.reduce_mean(entropy(logits=prediction, labels=y))
    # cost = tf.Print(cost, [cost], message="This is cost: ")
    optimizer = tf.train.AdamOptimizer(epsilon=.01, learning_rate=0.01, beta1=.8).minimize(cost)
    # optimizer = tf.Print(optimizer, [optimizer], message="This is optimizer: ")

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for k, v in enumerate(hm_epochs):
            start_time = time.time()
            for epoch in range(v):
                epoch_loss = 0
                for _ in range(int(mnist.train.num_examples / batch_size)):
                    epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                    epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                #                 print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            error.append(accuracy.eval({x: mnist.test.images.reshape(-1, n_chunks, chunk_size), y: mnist.test.labels}))
            time_taken.append(time.time() - start_time)
            print('Epochs::', v, '::Accuracy:', error[k])
    print(error)


train_neural_network(x)

print(time_taken)