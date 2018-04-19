from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#parameter 
learningRate = 0.005
numOfEpochs = 20
batchSize = 128
numOfBatches = int(mnist.train.num_examples/batchSize)

#network
x = tf.placeholder("float",[None,784])
y = tf.placeholder("float",[None,10])

weight1 = tf.Variable(tf.random_normal([784,512]))
weight2 = tf.Variable(tf.random_normal([512,512]))
weight3 = tf.Variable(tf.random_normal([512,10]))

bias1 = tf.Variable(tf.random_normal([512]))
bias2 = tf.Variable(tf.random_normal([512]))
bias3 = tf.Variable(tf.random_normal([10]))

layer1 = tf.nn.relu(tf.add(tf.matmul(x,weight1),bias1))
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,weight2),bias2))
output = tf.matmul(layer2,weight3) + bias3

#loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = output))
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(loss)

#initial
init = tf.global_variables_initializer()

#plot
epochSet = []
avgLossSet = []

#train 
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(numOfEpochs):
        avgLoss = 0
        for batch in range(numOfBatches):
            trainX, trainY = mnist.train.next_batch(batchSize)
            sess.run(optimizer,feed_dict={x:trainX,y:trainY})
            avgLoss += sess.run(loss,feed_dict={x:trainX,y:trainY})/numOfBatches
        print ("Epoch:",'%04d'%(epoch+1),"Loss=","{:.9f}".format(avgLoss))
        epochSet.append(epoch+1)
        avgLossSet.append(avgLoss)
    print("Training Complete!")
    
    #plot
    plt.plot(epochSet,avgLossSet)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    
    correctPrediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction,"float"))
    print ("Testing Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

