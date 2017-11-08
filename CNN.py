import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
from tensorflow.python import debug as tf_debug


n_classes = 10
#MNIST contains 10 classes in the dataset

batch_size = 128
#Increase, if GPU runs out of memory, the program will be slower

#Tensor placeholder, all variables in the program should be tensor variables/ placeholders.
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')
hm_epochs = [2,5,10,15,20,25,30]
error = []

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):

#     Our CNN will consist of 2 layers, where the first layer, will read an image as multiple(seven in this case)  5*5 pixels and produce 32 output variables,
#     the second layer reads the 32 variables as input and also interacts with it's kernel function g(in this case a 5*5 image), and produce 32+32 output variables
#     THe final convolution, will combine all the 7 5*5 convolution scans and produce an output for the output layer
#     https://github.com/iit-cs585/main/raw/34dcdbb0079f37206099e8ab5dd7e8c0d038f0bb/lec/l23/figs/all.png check the image for more information
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_conv3':tf.Variable(tf.random_normal([5,5,64,128])),
               'W_fc':tf.Variable(tf.random_normal([4*4*128,2048])),
               'out':tf.Variable(tf.random_normal([2048, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_conv3':tf.Variable(tf.random_normal([128])),
               'b_fc':tf.Variable(tf.random_normal([2048])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    #Below actions is basically h = max(Bias+ Weights*(h-1)
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    #relu is the rectified linear activation function
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    
    
    conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool2d(conv3)
    conv3Shape = conv3.get_shape().as_list()
    print(conv3Shape)
    fc = tf.reshape(conv3,[-1, conv3Shape[1]*conv3Shape[2]*conv3Shape[3]])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits= prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = .4
    with tf.Session(config=config) as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.initialize_all_variables())
        for k,v in enumerate(hm_epochs):
            print(k,v)
            for epoch in range(v):
                epoch_loss = 0
                for _ in range(int(mnist.train.num_examples/batch_size)):
                    epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

#                 print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#             good = []
#             temp = 0
#             #this step is only to perform batch wise testing. 6K would be sufficient for 4GB of memory
#             Test_batch_size=6000
#             test_set_size=len(mnist.test.labels)
#             for i in range(int(test_set_size/Test_batch_size)+1):
#                 testSet = mnist.test.next_batch(Test_batch_size)
#                 temp=accuracy.eval(feed_dict={ x: testSet[0], y: testSet[1], keep_prob: 1.0})
# #                 print(i,"::",temp)
#                 good.append(temp)
#             error.append(np.mean(good))
            error.append(accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
            print("Epoch::",v,"test accuracy::",error[k])
        print(error)

train_neural_network(x)