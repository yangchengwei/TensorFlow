from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#parameter
learningRate = 0.005
trainingIters = 100000
batchSize = 128
dropout = 0.2
numOfEpochs = 20
numOfBatches = int(mnist.train.num_examples/batchSize)
keepIn = 1.0-dropout

#training data
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keepProb = tf.placeholder(tf.float32)

#network variable
kernel1 = tf.Variable(tf.random_normal([5,5,1,32]))
bias1 = tf.Variable(tf.random_normal([32]))

kernel2 = tf.Variable(tf.random_normal([5,5,32,64]))
bias2 = tf.Variable(tf.random_normal([64]))

weight3 = tf.Variable(tf.random_normal([7*7*64,1024]))
bias3 = tf.Variable(tf.random_normal([1024]))

weight4 = tf.Variable(tf.random_normal([1024,10]))
bias4 = tf.Variable(tf.random_normal([10]))

#deep learn function
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#construct model
_X = tf.reshape(x,shape=[-1,28,28,1])
conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(_X,kernel1),bias1))
conv1 = max_pool_2x2(conv1)
conv1 = tf.nn.dropout(conv1,keepProb)

conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(conv1,kernel2),bias2))
conv2 = max_pool_2x2(conv2)
conv2 = tf.nn.dropout(conv2,keepProb)

layer3 = tf.reshape(conv2,shape=[-1,weight3.get_shape().as_list()[0]])
layer3 = tf.nn.relu(tf.add(tf.matmul(layer3,weight3),bias3))
layer3 = tf.nn.dropout(layer3,keepProb)

output = tf.add(tf.matmul(layer3,weight4),bias4)

#loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(cost)

#evaluate
correctPrediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction,tf.float32))

#initial
init = tf.global_variables_initializer()

epochSet = []
avgLossSet = []


#train
with tf.Session() as sess:
    sess.run(init)
    allStart = time.time()
    for epoch in range(numOfEpochs):
        start = time.time()
        avgLoss = 0
        for batch in range(numOfBatches):
            trainX, trainY = mnist.train.next_batch(batchSize)
            sess.run(optimizer,feed_dict={x:trainX,y:trainY,keepProb:keepIn})
            avgLoss += sess.run(cost,feed_dict={x:trainX,y:trainY,keepProb:keepIn})/numOfBatches
        epochSet.append(epoch+1)
        avgLossSet.append(avgLoss)
        end = time.time()
        print ("Epoch:",'%04d'%(epoch+1),"Loss=","{:.9f}".format(avgLoss),"Time=","{:.3f}".format(end-start))
    allEnd = time.time()
    print("Training Complete!","Time=","{:.4f}".format(allEnd-allStart))
    
    #plot
    plt.plot(epochSet,avgLossSet)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    print ("Testing Accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keepProb:1.0}))




