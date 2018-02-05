
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
from model_function import *

class WGAN_GP(object):
    model_name = "WGAN_GP"

    def __init__(self, sess, batchSize, numOfEpochs, learningRate,
                 restoreDir, informationDir, outputDir):
        self.sess = sess
        self.batchSize = batchSize
        self.numOfEpochs = numOfEpochs
        self.learningRate = learningRate
        self.restoreDir = restoreDir
        self.informationDir = informationDir
        self.outputDir = outputDir

        # FIXME: model parameter
        self.channel1 = 16
        self.channel2 = 32
        self.channel3 = 64
        self.channelIn = 1
        self.channelOut = 1
        self.imageHeight = 128
        self.imageWidth = 128
        self.lambd = 500       # The higher value, the more stable, but the slower convergence

        # FIXME: initial
        self.readList = []
        self.testData = -1
        self.interval = 9
        return


    def reset_training_data(self):
        self.readList.clear()
        # FIXME: self.readList=list(range(...))
        self.readList =      list(range(        1, 5*14*6+1))
        self.readList.extend(list(range( 8*14*6+1,13*14*6+1)))
        self.readList.extend(list(range(16*14*6+1,21*14*6+1)))
        self.readList.extend(list(range(24*14*6+1,29*14*6+1)))
        return


    def reset_testing_data(self):
        self.readList.clear()
        # FIXME: self.readList=list(range(...))
        self.readList=list(range(1,2437))
        self.readList.reverse()
        return


    def read_image(self, size, rand=True):
        inputImage = np.empty(shape=[0, self.imageHeight, self.imageWidth, self.channelIn])
        outputImage = np.empty(shape=[0, self.imageHeight, self.imageWidth, self.channelOut])
        readNow = 0
        for i in range(size):
            if (len(self.readList)==0):
                break
            if rand :
                readNow = random.choice(self.readList)
                self.readList.remove(readNow)
            else :
                readNow = self.readList.pop()
            # FIXME: input image
            im = io.imread('./../downsize_input/' + '%04d.bmp'%(readNow), as_grey=True)
            im = im.reshape( 1, self.imageHeight, self.imageWidth, self.channelIn) /255
            inputImage = np.append(inputImage, im, axis=0)

            im = io.imread('./../downsize_output/' + '%04d.bmp'%(readNow), as_grey=True)
            im = im.reshape( 1, self.imageHeight, self.imageWidth, self.channelOut) /255
            outputImage = np.append(outputImage, im, axis=0)
        return inputImage, outputImage, readNow


    def generator(self, input_tensor, scope='generator'):
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            #FIXME:
            out = conv2d(input_tensor, self.channelIn, self.channel1, scope='conv1', s=2, addBias=True, activated=True)
            out = conv2d(out, self.channel1, self.channel2, scope='conv2', s=2, addBias=True, batchNorm=True, activated=True)
            out = conv2d(out, self.channel2, self.channel3, scope='conv3', s=2, addBias=True, activated=True)
            out = deconv2d(out, self.channel3, self.channel2, scope='deconv3', s=2, addBias=True, batchNorm=True, activated=True)
            out = deconv2d(out, self.channel2, self.channel1, scope='deconv2', s=2, addBias=True, activated=True)
            out = deconv2d(out, self.channel1, self.channelOut, scope='deconv1', s=2, addBias=True)
            return out


    def discriminator(self, input_tensor, scope='discriminator'):
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            #FIXME:
            out = conv2d(input_tensor, self.channelOut, self.channel1, scope='conv1', s=2, addBias=True, activated=True)
            out = max_pool(out)
            out = conv2d(out, self.channel1, self.channel2, scope='conv2', s=2, addBias=True, batchNorm=True, activated=True)
            out = max_pool(out)
            out = flatten(out) #64 * channel2
            out = linear(out, 2048, 512, scope='linear1', addBias=True, activated=True)
            out = linear(out, 512, 64, scope='linear2', addBias=True, activated=True)
            out = linear(out, 64, 1, scope='linear3', addBias=True)
            return out


    def build_model(self):
        self.IP = tf.placeholder(tf.float32,[self.batchSize, self.imageHeight, self.imageWidth, self.channelIn])
        self.GT = tf.placeholder(tf.float32,[self.batchSize, self.imageHeight, self.imageWidth, self.channelOut])
        self.LR = tf.placeholder(tf.float32)

        self.G_fake = self.generator(self.IP)
        self.D_fake = self.discriminator(self.G_fake)
        self.D_real = self.discriminator(self.GT)

        D_loss_fake = tf.reduce_mean(self.D_fake)
        D_loss_real = tf.reduce_mean(self.D_real)

        self.D_loss = - ( D_loss_real-D_loss_fake )
        self.G_loss = - D_loss_fake

        """ Gradient Penalty """
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        alpha = tf.random_uniform(shape=self.GT.get_shape(), minval=0., maxval=1.)
        differences = self.G_fake - self.GT # This is different from MAGAN
        interpolates = self.GT + (alpha * differences)
        D_inter = self.discriminator(interpolates)
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.D_loss += self.lambd * gradient_penalty

        """ Training """
        # divide trainable variables into a group for D and a group for G
        T_vars = tf.trainable_variables()
        D_vars = [var for var in T_vars if 'discriminator' in var.name]
        G_vars = [var for var in T_vars if 'generator' in var.name]

        self.D_opti = tf.train.RMSPropOptimizer(self.LR).minimize(self.D_loss, var_list=D_vars)
        self.G_opti = tf.train.RMSPropOptimizer(self.LR*5).minimize(self.G_loss, var_list=G_vars)

        self.saver = tf.train.Saver()
        tf.get_variable_scope().reuse_variables()
        return


    def train(self):
        epochSet = []
        D_avgLossSet = []
        G_avgLossSet = []
        tempRate = self.learningRate

        print('Training Start!')

        allStart = time.time()
        for epoch in range(1, self.numOfEpochs+1):

            self.reset_training_data()

            D_sumLoss = 0
            G_sumLoss = 0
            batch = 0

            epochStart = time.time()
            while (len(self.readList) != 0):
                trainX, trainY, readNow = self.read_image(self.batchSize)
                _, D_batchLoss, _, G_batchLoss = self.sess.run([self.D_opti, self.D_loss, self.G_opti, self.G_loss],
                                             feed_dict={self.IP : trainX,
                                                        self.GT : trainY,
                                                        self.LR : tempRate})
                D_sumLoss += D_batchLoss
                G_sumLoss += G_batchLoss
                batch = batch + 1

            self.readList = [5*14*6+1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            trainX, trainY, readNow = self.read_image(self.batchSize)
            image = self.sess.run(self.G_fake, feed_dict={self.IP : trainX})
            image = image[0].reshape(self.imageHeight,self.imageWidth)
            image[image >=0.5]=1
            image[image < 0.5]=0
            io.imsave(self.outputDir + '%04d.bmp'%(epoch),image)
            
            epochEnd = time.time()
            

            D_avgLoss = D_sumLoss / batch
            G_avgLoss = G_sumLoss / batch

            print('\nEpoch: %04d D_avgLoss= %.9f G_avgLoss= %.9f Time= %.9f\n'%(epoch, D_avgLoss, G_avgLoss, epochEnd-epochStart))
            cmdFile = open(self.informationDir+'cmd.txt','a')
            cmdFile.write('Epoch: %04d D_avgLoss= %.9f G_avgLoss= %.9f Time= %.9f \n'%(epoch, D_avgLoss, G_avgLoss, epochEnd-epochStart))
            cmdFile.close()

            epochSet.append(epoch)
            D_avgLossSet.append(D_avgLoss)
            G_avgLossSet.append(G_avgLoss)

            if (epoch == self.numOfEpochs / 2):
                tempRate = tempRate / 10

                print('Change learningRate to %f'%(tempRate))
                cmdFile = open(self.informationDir+'cmd.txt','a')
                cmdFile.write('Change learningRate to %f\n'%(tempRate))
                cmdFile.close()

                # plot D
                plt.plot(epochSet,D_avgLossSet)
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.savefig(self.informationDir+'D_loss%d.png'%(epoch), dpi=100)
                plt.show()
                
                # plot G
                plt.plot(epochSet,G_avgLossSet)
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.savefig(self.informationDir+'G_loss%d.png'%(epoch), dpi=100)
                plt.show()

                # plot init
                epochSet = []
                D_avgLossSet = []
                G_avgLossSet = []

        allEnd = time.time()

        print('Training Complete! Time= %.9f'%(allEnd-allStart))
        cmdFile = open(self.informationDir+'cmd.txt','a')
        cmdFile.write('Training Complete! Time= %.9f\n'%(allEnd-allStart))
        cmdFile.close()

        # plot D
        plt.plot(epochSet,D_avgLossSet)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.informationDir+'D_loss%d.png'%(epoch), dpi=100)
        plt.show()
        
        # plot G
        plt.plot(epochSet,G_avgLossSet)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.informationDir+'G_loss%d.png'%(epoch), dpi=100)
        plt.show()

        self.saver.save(self.sess, self.informationDir+'model%d.ckpt'%(self.interval))
        return


    def test(self):
        firstFrame = True
        gt1 = np.empty([1, self.imageHeight, self.imageWidth, 0])

        dataSet = []
        sumDiffDC = 0
        accuracySetDC = []
        fileDC = open(self.informationDir + 'accuracyDC.txt','a')

        self.reset_testing_data()

        while len(self.readList) != 0 :
            trainX, trainY, readNow = self.read_image(1, randomChoice=False)
            if firstFrame:
                io.imsave(self.outputDir+'%04d.jpg'%(readNow-self.interval), trainX[0,:,:,2])
                firstFrame = False
            else:
                trainX[0,:,:,2] = gt1[0,:,:,0]

            output = self.sess.run(self.PD, feed_dict={self.IP : trainX})

            output[output> 0.5]=1
            output[output<=0.5]=0
            output = output[0,:,:,0]

            output = self.image_to_original_size(output)

            gt1 = output.reshape( 1, self.imageHeight, self.imageWidth, 1)
            OP  = output.reshape(self.imageHeight,self.imageWidth)

            io.imsave(self.outputDir+'%04d.jpg'%(readNow), OP)

            GT  = np.reshape(trainY, [self.imageHeight,self.imageWidth])
            GT[GT> 0.5]=1
            GT[GT<=0.5]=0
            areaOP = np.sum(OP)
            areaGT = np.sum(GT)
            accuracyDC = 1 - ( np.sum(np.abs(GT-OP)) / (areaOP + areaGT) )
            dataSet.append(readNow)
            accuracySetDC.append(accuracyDC*100)
            sumDiffDC += accuracyDC
            fileDC.write('Data: %04d Accuracy= %.9f \n'%(readNow,accuracyDC))

        fileDC.write('%02d Average_Accuracy: %.9f \n'%(self.testData,(sumDiffDC/self.testSize)))
        fileDC.close()

        plt.bar(dataSet,accuracySetDC)
        plt.xlabel('data')
        plt.ylabel('accuracy(%)')
#        plt.ylim(90,100)
        plt.savefig(self.informationDir + 'accuracyDC_%04d.png'%(self.testData))
        plt.close()
        return


    def init(self) :
        tf.global_variables_initializer().run()
        return


    def save(self, modelName) :
        self.saver.save(self.sess, self.informationDir + modelName)
        return


    def restore(self, modelName):
        self.saver.restore(self.sess, self.restoreDir + modelName)
        return

