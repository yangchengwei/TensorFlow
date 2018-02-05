
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
from model_function import *

class U_Net(object):
    model_name = "U_Net"

    def __init__(self, sess, batchSize, numOfEpochs, learningRate,
                 restoreDir, informationDir, outputDir):
        self.sess = sess
        self.batchSize = batchSize
        self.numOfEpochs = numOfEpochs
        self.learningRate = learningRate
        self.restoreDir = restoreDir
        self.informationDir = informationDir
        self.outputDir = outputDir

        #self.dataSize = 9486
        self.numOfKernel1 = 16
        self.numOfKernel2 = 32
        self.numOfKernel3 = 64
        self.numOfKernel4 = 64
        self.inputChannel = 3
        self.outputChannel = 1
        self.imageHeight = 360
        self.imageWidth = 800
        self.interval = 9

        self.whoseGT = 'mo/'

        self.readList = []
        self.testData = -1

        return

    def reset_training_data(self):
        self.readList.clear()
        '''normal04'''
        self.readList =      list(range(   0,  408-self.interval))
        self.readList.extend(list(range( 408,  816-self.interval)))
        self.readList.extend(list(range( 816, 1224-self.interval)))
        self.readList.extend(list(range(1224, 1632-self.interval)))
        '''normal05'''
        self.readList.extend(list(range(1632, 2023-self.interval)))
        self.readList.extend(list(range(2023, 2414-self.interval)))
        self.readList.extend(list(range(2414, 2805-self.interval)))
        self.readList.extend(list(range(2805, 3196-self.interval)))
        '''normal06'''
        self.readList.extend(list(range(3196, 3604-self.interval)))
        self.readList.extend(list(range(3604, 3995-self.interval)))
        self.readList.extend(list(range(3995, 4386-self.interval)))
        self.readList.extend(list(range(4386, 4777-self.interval)))
        '''patient01'''
        self.readList.extend(list(range(6341, 6732-self.interval)))
        self.readList.extend(list(range(6732, 7123-self.interval)))
        self.readList.extend(list(range(7123, 7531-self.interval)))
        self.readList.extend(list(range(7531, 7922-self.interval)))

        return


    def reset_testing_data(self):
        self.readList.clear()
        data_list = [0,408,816,1224,1632,
                     2023,2414,2805,3196,
                     3604,3995,4386,4777,
                     5168,5559,5950,6341,
                     6732,7123,7531,7922,
                     8313,8704,9095,9486]
        self.readList = list(range(data_list[self.testData], data_list[self.testData+1]-self.interval, self.interval))
        self.readList.reverse()
        self.testSize = len(self.readList)
        return


    def read_image(self, size, randomChoice=True):
        inputImage = np.empty(shape=[0, self.imageHeight, self.imageWidth, self.inputChannel])
        outputImage = np.empty(shape=[0, self.imageHeight, self.imageWidth, self.outputChannel])
        readNow = 0
        for i in range(size):
            if (len(self.readList)==0):
                print('Read all image!')
                break
            if randomChoice :
                readNow = random.choice(self.readList)
                self.readList.remove(readNow)
            else :
                readNow = self.readList.pop()

            im1 = io.imread('./../data2_input/' + '%04d.jpg'%(readNow), as_grey=True)
            im2 = io.imread('./../data2_input/' + '%04d.jpg'%(readNow + self.interval), as_grey=True)
            gt1 = io.imread('./../data2_output/' + self.whoseGT + '%04d.jpg'%(readNow), as_grey=True)

            im1 = im1.reshape( 1, self.imageHeight, self.imageWidth, 1)
            im2 = im2.reshape( 1, self.imageHeight, self.imageWidth, 1)
            gt1 = gt1.reshape( 1, self.imageHeight, self.imageWidth, 1)

            im = np.empty(shape=[1, self.imageHeight, self.imageWidth, 0])
            im = np.append(im,im1,axis=3)
            im = np.append(im,im2,axis=3)
            im = np.append(im,gt1,axis=3)
            inputImage = np.append(inputImage, im, axis=0)

            im = io.imread('./../data2_output/' + self.whoseGT + '%04d.jpg'%(readNow + self.interval), as_grey=True)
            im = im.reshape( 1, self.imageHeight, self.imageWidth, self.outputChannel)
            outputImage = np.append(outputImage, im, axis=0)
        return inputImage, outputImage, readNow + self.interval


    def model(self, input_tensor, scope='model'):
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            out = conv2d( input_tensor, self.inputChannel, self.numOfKernel1, scope='conv1_1',
                         addBias=False, padding='VALID')
            out1 = conv2d( out, self.numOfKernel1, self.numOfKernel1, scope='conv1_2',
                          addBias=False, padding='VALID')
            out = max_pool(out1)
            
            out = conv2d(out, self.numOfKernel1, self.numOfKernel2, scope='conv2_1',
                         addBias=False, padding='VALID')
            out2 = conv2d(out, self.numOfKernel2, self.numOfKernel2, scope='conv2_2',
                          addBias=False, padding='VALID')
            out = max_pool(out2)
            
            out = conv2d(out, self.numOfKernel2, self.numOfKernel3, scope='conv3_1',
                         addBias=False, padding='VALID')
            out3 = conv2d(out, self.numOfKernel3, self.numOfKernel3, scope='conv3_2',
                          addBias=False, padding='VALID')
            out = max_pool(out3)
            
            out = conv2d(out, self.numOfKernel3, self.numOfKernel4, scope='conv4_1',
                         addBias=False, padding='VALID')
            out = conv2d(out, self.numOfKernel4, self.numOfKernel4, scope='conv4_2',
                         addBias=False, padding='VALID')
            out = conv2d_transpose(out, self.numOfKernel4, self.numOfKernel4//2, scope='deconv4',
                                   addBias=False)
            
            out = crop_and_concat(out3, out)
            out = conv2d(out, self.numOfKernel4//2+self.numOfKernel3, self.numOfKernel3, scope='conv3_3',
                         addBias=False, padding='VALID')
            out = conv2d(out, self.numOfKernel3, self.numOfKernel3, scope='conv3_4',
                         addBias=False, padding='VALID')
            out = conv2d_transpose(out, self.numOfKernel3, self.numOfKernel3//2, scope='deconv3',
                                   addBias=False)
            
            out = crop_and_concat(out2, out)
            out = conv2d(out, self.numOfKernel3//2+self.numOfKernel2, self.numOfKernel2,
                         scope='conv2_3', addBias=False, padding='VALID')
            out = conv2d(out, self.numOfKernel2, self.numOfKernel2, scope='conv2_4',
                         addBias=False, padding='VALID')
            out = conv2d_transpose(out, self.numOfKernel2, self.numOfKernel2//2, scope='deconv2',
                                   addBias=False)
            
            out = crop_and_concat(out1, out)
            out = conv2d(out, self.numOfKernel2//2+self.numOfKernel1, self.numOfKernel1, scope='conv1_3',
                         addBias=False, padding='VALID')
            out = conv2d(out, self.numOfKernel1, self.numOfKernel1, scope='conv1_4',
                         addBias=False, padding='VALID')
            out = conv2d(out, self.numOfKernel1, self.outputChannel, scope='conv1_5',
                         addBias=False, activated=False)
            
            return out


    def build_model(self):
        self.IP = tf.placeholder(tf.float32,[None, self.imageHeight, self.imageWidth, self.inputChannel])
        self.GT = tf.placeholder(tf.float32,[None, self.imageHeight, self.imageWidth, self.outputChannel])
        self.LR = tf.placeholder(tf.float32)

        self.PD = self.model(self.IP)

        self.loss = tf.nn.l2_loss(tf.abs(crop(self.GT,self.PD)-self.PD))
        self.opti = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

        self.saver = tf.train.Saver()

        tf.get_variable_scope().reuse_variables()
        return

    def train(self):
        epochSet = []
        avgLossSet = []
        learningRate = self.learningRate

        print('Training Start!')

        allStart = time.time()
        for epoch in range(1, self.numOfEpochs+1):

            self.reset_training_data()

            sumLoss = 0
            batch = 0

            epochStart = time.time()
            while (len(self.readList) != 0):
                trainX, trainY, readNow = self.read_image(self.batchSize)
                _, batchLoss = self.sess.run([self.opti, self.loss],
                                             feed_dict={self.IP : trainX,
                                                        self.GT : trainY,
                                                        self.LR : learningRate})
                sumLoss += batchLoss
                batch = batch + 1

                if ((batch)%20 == 0):
                    print('Batch: %04d BatchLoss: %.9f'%(batch,batchLoss))

            epochEnd = time.time()

            avgLoss = sumLoss / batch

            print('Epoch: %04d Loss= %.9f Time= %.9f'%(epoch,avgLoss,epochEnd-epochStart))
            cmdFile = open(self.informationDir+'cmd.txt','a')
            cmdFile.write('Epoch: %04d Loss= %.9f Time= %.9f\n'%(epoch,avgLoss,epochEnd-epochStart))
            cmdFile.close()

            epochSet.append(epoch)
            avgLossSet.append(avgLoss)

            if (epoch == self.numOfEpochs / 2):
                self.learningRate = self.learningRate / 10

                print('Change learningRate to %f'%(self.learningRate))
                cmdFile = open(self.informationDir+'cmd.txt','a')
                cmdFile.write('Change learningRate to %f\n'%(self.learningRate))
                cmdFile.close()

                # plot
                plt.plot(epochSet,avgLossSet)
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.savefig(self.informationDir+'loss%d.png'%(epoch), dpi=100)
                plt.show()

                # plot init
                epochSet = []
                avgLossSet = []

        allEnd = time.time()

        print('Training Complete! Time= %.9f'%(allEnd-allStart))
        cmdFile = open(self.informationDir+'cmd.txt','a')
        cmdFile.write('Training Complete! Time= %.9f\n'%(allEnd-allStart))
        cmdFile.close()

        # plot
        plt.plot(epochSet,avgLossSet)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(self.informationDir+'loss%d.png'%(epoch), dpi=100)
        plt.show()

        # save and restore final model
        self.saver.save(self.sess, self.informationDir+'model%d.ckpt'%(self.interval))

        return
    
    def image_to_original_size(self,img):
        y,x = img.shape
        ori_image = np.zeros([self.imageHeight, self.imageWidth], dtype = np.float32)
        startx = self.imageWidth//2-(x//2)
        starty = self.imageHeight//2-(y//2)
        ori_image[starty:starty+y,startx:startx+x] = img
        return ori_image
        
        
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

    def save(self, modelName) :
        self.saver.save(self.sess, self.informationDir + modelName)

    def restore(self, modelName):
        self.saver.restore(self.sess, self.restoreDir + modelName)