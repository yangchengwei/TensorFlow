
import cv2
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model_function import *

class U_Net(object):
    model_name = "U_Net"

    def __init__(self, sess=None, batchSize=1, numOfEpochs=0, learningRate=0, max_to_keep=5,
                 restoreDir='./', informationDir='./', outputDir='./'):
        self.sess = sess
        self.batchSize = batchSize
        self.numOfEpochs = numOfEpochs
        self.learningRate = learningRate
        self.max_to_keep = max_to_keep
        self.restoreDir = restoreDir
        self.informationDir = informationDir
        self.outputDir = outputDir

        self.numOfKernel1 = 16
        self.numOfKernel2 = 32
        self.numOfKernel3 = 64
        self.numOfKernel4 = 128
        self.inputChannel = 2
        self.outputChannel = 1
        self.imageHeight = 256
        self.imageWidth = 512

        self.readList = []
        self.training_list_input = []
        self.training_list_output = []
        self.testData = -1
        self.interval = 20
        self.dila_iter = 8
        return


    def reset_training_data(self):
        self.readList.clear()
        '''normal04'''
        self.readList.extend(list(range(   0+self.interval,  408-self.interval)))
        self.readList.extend(list(range( 408+self.interval,  816-self.interval)))
#        self.readList.extend(list(range( 816+self.interval, 1224-self.interval)))
#        self.readList.extend(list(range(1224+self.interval, 1632-self.interval)))
        '''normal05'''
        self.readList.extend(list(range(1632+self.interval, 2023-self.interval)))
        self.readList.extend(list(range(2023+self.interval, 2414-self.interval)))
#        self.readList.extend(list(range(2414+self.interval, 2805-self.interval)))
#        self.readList.extend(list(range(2805+self.interval, 3196-self.interval)))
        '''normal06'''
        self.readList.extend(list(range(3196+self.interval, 3604-self.interval)))
        self.readList.extend(list(range(3604+self.interval, 3995-self.interval)))
#        self.readList.extend(list(range(3995+self.interval, 4386-self.interval)))
#        self.readList.extend(list(range(4386+self.interval, 4777-self.interval)))
        '''normal08'''
        self.readList.extend(list(range(4777+self.interval, 5168-self.interval)))
        self.readList.extend(list(range(5168+self.interval, 5559-self.interval)))
#        self.readList.extend(list(range(5559+self.interval, 5950-self.interval)))
#        self.readList.extend(list(range(5950+self.interval, 6341-self.interval)))
        '''patient01'''
        self.readList.extend(list(range(6341+self.interval, 6732-self.interval)))
        self.readList.extend(list(range(6732+self.interval, 7123-self.interval)))
#        self.readList.extend(list(range(7123+self.interval, 7531-self.interval)))
#        self.readList.extend(list(range(7531+self.interval, 7922-self.interval)))
        '''patient02'''
        self.readList.extend(list(range(7922+self.interval, 8313-self.interval)))
        self.readList.extend(list(range(8313+self.interval, 8704-self.interval)))
#        self.readList.extend(list(range(8704+self.interval, 9095-self.interval)))
#        self.readList.extend(list(range(9095+self.interval, 9486-self.interval)))
        return
        
        
    def reset_validation_data(self):
        self.readList.clear()
        '''normal04'''
        self.readList.extend(list(range( 816+self.interval, 1224-self.interval)))
        '''normal05'''
        self.readList.extend(list(range(2414+self.interval, 2805-self.interval)))
        '''normal06'''
        self.readList.extend(list(range(3995+self.interval, 4386-self.interval)))
        '''normal08'''
        self.readList.extend(list(range(5559+self.interval, 5950-self.interval)))
        '''patient01'''
        self.readList.extend(list(range(7123+self.interval, 7531-self.interval)))
        '''patient02'''
        self.readList.extend(list(range(8704+self.interval, 9095-self.interval)))
        return


    def reset_testing_data(self):
        self.readList.clear()
        data_list = [0,408,816,1224,1632,
                     2023,2414,2805,3196,
                     3604,3995,4386,4777,
                     5168,5559,5950,6341,
                     6732,7123,7531,7922,
                     8313,8704,9095,9486]
        self.readList = list(range(data_list[self.testData]+self.interval, data_list[self.testData+1], self.interval))
        self.readList.reverse()
        self.testSize = len(self.readList)
        return


    def read_all_data(self):
        print('Start read_all_data !')
        self.training_list_input.clear()
        self.training_list_output.clear()
        for i in range(9486):
            print(i)
            image_input = cv2.imread('./../data2_input/%04d.jpg'%(i), cv2.IMREAD_GRAYSCALE)
            image_output = cv2.imread('./../data2_output/mo/%04d.jpg'%(i), cv2.IMREAD_GRAYSCALE)
            
            image_input = cv2.resize(image_input, dsize=(self.imageWidth,self.imageHeight))
            image_output = cv2.resize(image_output, dsize=(self.imageWidth,self.imageHeight))
            
            image_input = np.array(image_input, dtype=np.uint8)
            image_output = np.array(image_output, dtype=np.uint8)
            
            self.training_list_input.append(image_input)
            self.training_list_output.append(image_output)
        print('End read_all_data !')
        return


    def read_image(self, size, randomChoice=True, augment=True):
        inputImage = np.empty(shape=[0, self.imageHeight, self.imageWidth, self.inputChannel], dtype = np.float32)
        outputImage = np.empty(shape=[0, self.imageHeight, self.imageWidth, self.outputChannel], dtype = np.float32)
        readNow = 0
        for i in range(size):
            if (len(self.readList)==0):
                break
            if randomChoice :
                readNow = random.choice(self.readList)
                self.readList.remove(readNow)
            else :
                readNow = self.readList.pop()
            
            if augment:
                degree = random.randint( -15, 15 ) #-15,15
                M = cv2.getRotationMatrix2D((256,70), degree, 1.0)
                
                im2 = np.copy(self.training_list_input[readNow])
                im2 = im2 * (random.random() + 0.5) #+0.5
                im2[im2>255] = 255
                gt1 = self.training_list_output[readNow-self.interval+random.randint(0,2*self.interval)]
#                gt1 = cv2.dilate(gt1, np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8), iterations=random.randint(self.dila_iter-4,self.dila_iter+4))
                gt2 = self.training_list_output[readNow]
                
                im2 = cv2.warpAffine(im2, M, (512, 256))
                gt1 = cv2.warpAffine(gt1, M, (512, 256))
                gt2 = cv2.warpAffine(gt2, M, (512, 256))
            else :
                im2 = self.training_list_input[readNow]
                gt1 = self.training_list_output[readNow-self.interval]
#                gt1 = cv2.dilate(gt1, np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8), iterations=self.dila_iter)
                gt2 = self.training_list_output[readNow]
                
            im = np.empty(shape=[1, self.imageHeight, self.imageWidth, 0])
            im = np.append(im,im2.reshape( 1, self.imageHeight, self.imageWidth, 1),axis=3)
            im = np.append(im,gt1.reshape( 1, self.imageHeight, self.imageWidth, 1),axis=3)
            inputImage = np.append(inputImage, im, axis=0)

            outputImage = np.append(outputImage, gt2.reshape( 1, self.imageHeight, self.imageWidth, 1), axis=0)
        inputImage = inputImage / 255 * 2 - 1
        outputImage = outputImage / 255 * 2 - 1
        return inputImage, outputImage, readNow


    def model(self, input_tensor, scope='model'):
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            out = conv2d(input_tensor, self.inputChannel, self.numOfKernel1, scope='conv1_1', padding='SAME',
                         batchNorm=True, activated=True)
            out1 = conv2d(out, self.numOfKernel1, self.numOfKernel1, scope='conv1_2', padding='SAME',
                         batchNorm=True, activated=True)
            out = max_pool(out1)
            
            out = conv2d(out, self.numOfKernel1, self.numOfKernel2, scope='conv2_1', padding='SAME',
                         batchNorm=True, activated=True)
            out2 = conv2d(out, self.numOfKernel2, self.numOfKernel2, scope='conv2_2', padding='SAME',
                         batchNorm=True, activated=True)
            out = max_pool(out2)
            
            out = conv2d(out, self.numOfKernel2, self.numOfKernel3, scope='conv3_1', padding='SAME',
                         batchNorm=True, activated=True)
            out3 = conv2d(out, self.numOfKernel3, self.numOfKernel3, scope='conv3_2', padding='SAME',
                         batchNorm=True, activated=True)
            out = max_pool(out3)
            
            out = conv2d(out, self.numOfKernel3, self.numOfKernel4, scope='conv4_1', padding='SAME',
                         batchNorm=True, activated=True)
            out = conv2d(out, self.numOfKernel4, self.numOfKernel4, scope='conv4_2', padding='SAME',
                         batchNorm=True, activated=True)
            out = deconv2d(out, self.numOfKernel4, self.numOfKernel4//2, s=2, scope='deconv4',
                                   activated=True)
            
            out = crop_and_concat(out3, out)
            out = conv2d(out, self.numOfKernel4//2+self.numOfKernel3, self.numOfKernel3, scope='conv3_3', padding='SAME',
                         batchNorm=True, activated=True)
            out = conv2d(out, self.numOfKernel3, self.numOfKernel3, scope='conv3_4', padding='SAME',
                         batchNorm=True, activated=True)
            out = deconv2d(out, self.numOfKernel3, self.numOfKernel3//2, s=2, scope='deconv3',
                                   activated=True)
            
            out = crop_and_concat(out2, out)
            out = conv2d(out, self.numOfKernel3//2+self.numOfKernel2, self.numOfKernel2, scope='conv2_3', padding='SAME',
                         batchNorm=True, activated=True)
            out = conv2d(out, self.numOfKernel2, self.numOfKernel2, scope='conv2_4', padding='SAME',
                         batchNorm=True, activated=True)
            out = deconv2d(out, self.numOfKernel2, self.numOfKernel2//2, s=2, scope='deconv2', 
                                   activated=True)
            
            out = crop_and_concat(out1, out)
            out = conv2d(out, self.numOfKernel2//2+self.numOfKernel1, self.numOfKernel1, scope='conv1_3', padding='SAME',
                         activated=True)
            out = conv2d(out, self.numOfKernel1, self.numOfKernel1, scope='conv1_4', padding='SAME',
                         activated=True)
            out = conv2d(out, self.numOfKernel1, self.outputChannel, scope='conv1_5')

#        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
#            out = dense_block(input_tensor, 3, 4, scope='block1')
#            out = dense_block(out, 3, 4, scope='block2')
#            out = conv2d(out, tf.shape(out)[3], 1, scope='conv', addBias=True)

#        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
#            out = conv2d( input_tensor, self.inputChannel, self.numOfKernel1, scope='conv1',
#                         s=2, addBias=True, activated=True)
#            out = conv2d( out, self.numOfKernel1, self.numOfKernel2, scope='conv2',
#                         s=2, addBias=True, batchNorm=True, activated=True)
#            out = conv2d( out, self.numOfKernel2, self.numOfKernel3, scope='conv3',
#                         s=2, addBias=True, activated=True)
#            out = conv2d( out, self.numOfKernel3, self.numOfKernel4, scope='conv4',
#                         s=2, addBias=True, batchNorm=True, activated=True)
#            out = conv2d( out, self.numOfKernel4, self.numOfKernel4, scope='conv5',
#                         addBias=True, activated=True)
#            out = deconv2d( out, self.numOfKernel4, self.numOfKernel3, scope='deconv4',
#                         s=2, addBias=True, activated=True)
#            out = deconv2d( out, self.numOfKernel3, self.numOfKernel2, scope='deconv3',
#                         s=2, addBias=True, activated=True)
#            out = deconv2d( out, self.numOfKernel2, self.numOfKernel1, scope='deconv2',
#                         s=2, addBias=True, activated=True)
#            out = deconv2d( out, self.numOfKernel1, self.outputChannel, scope='deconv1',
#                         s=2, addBias=True)
        return out


    def build_model(self):
        print('Start build_model !')
        self.IP = tf.placeholder(tf.float32,[None, self.imageHeight, self.imageWidth, self.inputChannel])
        self.GT = tf.placeholder(tf.float32,[None, self.imageHeight, self.imageWidth, self.outputChannel])
        self.LR = tf.placeholder(tf.float32)

        self.PD = self.model(self.IP)

        self.loss = tf.nn.l2_loss(tf.add(self.GT,-self.PD))
        self.opti = tf.train.RMSPropOptimizer(self.LR).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep = self.max_to_keep)

        tf.get_variable_scope().reuse_variables()
        return


    def train(self):
        print('Start train !')
        epochSet = []
        avgLossSet = []
        valLossSet = []
        trainRate = self.learningRate
        self.min_valid_loss = 100000
        
        allStart = time.time()
        for epoch in range(1, self.numOfEpochs+1):
            epochStart = time.time()
            
            '''training'''
            
            self.reset_training_data()
            sumLoss_train = 0
            batch_train = 0
            while (len(self.readList) != 0):
                trainX, trainY, readNow = self.read_image(self.batchSize)
                _, batchLoss = self.sess.run([self.opti, self.loss], feed_dict={self.IP : trainX,
                                                                                self.GT : trainY,
                                                                                self.LR : trainRate})
                sumLoss_train += batchLoss
                batch_train = batch_train + 1
                if ((batch_train)%20 == 0):
                    print('Batch: %04d BatchLoss: %.9f'%(batch_train,batchLoss))
                    
            '''validation'''
            
            self.reset_validation_data()
            sumLoss_valid = 0
            batch_valid = 0
            while (len(self.readList) != 0):
                trainX, trainY, readNow = self.read_image(self.batchSize,randomChoice=False,augment=False)
                batchLoss = self.sess.run(self.loss, feed_dict={self.IP : trainX,
                                                                self.GT : trainY,
                                                                self.LR : trainRate})
                sumLoss_valid += batchLoss
                batch_valid = batch_valid + 1
                    
            avgLoss_train = sumLoss_train / batch_train
            avgLoss_valid = sumLoss_valid / batch_valid
            
            epochEnd = time.time()

            print('Epoch: %04d train_loss= %.9f valid_loss= %.9f Time= %.9f'
                  %(epoch,avgLoss_train,avgLoss_valid,epochEnd-epochStart))
            cmdFile = open(self.informationDir+'cmd.txt','a')
            cmdFile.write('Epoch: %04d train_Loss= %.9f valid_loss= %.9f Time= %.9f \n'
                          %(epoch,avgLoss_train,avgLoss_valid,epochEnd-epochStart))
            cmdFile.close()

            epochSet.append(epoch)
            avgLossSet.append(avgLoss_train)
            valLossSet.append(avgLoss_valid)
            
            if (self.min_valid_loss > avgLoss_valid):
                self.min_valid_loss = avgLoss_valid
                
                plt.plot(epochSet,avgLossSet)
                plt.plot(epochSet,valLossSet, color='red', linestyle='--')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.ylim(0,5000)
                plt.savefig(self.informationDir+'loss%d.png'%(epoch), dpi=100)
                plt.show()
                
                self.save('model%d.ckpt'%(epoch))
            elif (epoch%10 == 0):
                self.save('model%d.ckpt'%(epoch))
                
            # plot
            plt.plot(epochSet,avgLossSet)
            plt.plot(epochSet,valLossSet, color='red', linestyle='--')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.ylim(0,5000)
            plt.savefig(self.informationDir+'loss_final.png', dpi=100)
            plt.show()
            
            # save and restore final model
            self.save('model_final.ckpt')
            
        allEnd = time.time()

        print('Training Complete! Time= %.9f'%(allEnd-allStart))
        cmdFile = open(self.informationDir+'cmd.txt','a')
        cmdFile.write('Training Complete! Time= %.9f\n'%(allEnd-allStart))
        cmdFile.close()
        return


#    def image_to_original_size(self,img):
#        y,x = img.shape
#        ori_image = np.zeros([self.imageHeight, self.imageWidth], dtype = np.float32)
#        startx = self.imageWidth//2-(x//2)
#        starty = self.imageHeight//2-(y//2)
#        ori_image[starty:starty+y,startx:startx+x] = img
#        return ori_image
        
        
    def test(self):
        roi = np.empty([1, self.imageHeight, self.imageWidth, 0])

        dataSet = []
        sumDiffDC = 0
        accuracySetDC = []
        fileDC = open(self.outputDir + 'accuracyDC.txt','a')
        fileAvgDC = open(self.outputDir + 'accuracyAvgDC.txt','a')
        first_frame = True

        self.reset_testing_data()

        while len(self.readList) != 0 :
            trainX, trainY, readNow = self.read_image(1, randomChoice=False, augment=False)
            if first_frame:
                cv2.imwrite(self.outputDir+'%04d.jpg'%(readNow-self.interval), (trainX[0,:,:,1]+1)/2*255)
                first_frame = False
            else:
                trainX[0,:,:,1] = roi[0,:,:,0]

            OP = self.sess.run(self.PD, feed_dict={self.IP : trainX})
            
            OP[OP> 0]=255
            OP[OP<=0]=0
            OP = np.array(OP[0,:,:,0], dtype = np.uint8)
            cv2.imwrite(self.outputDir+'%04d.jpg'%(readNow), OP)

            roi = cv2.dilate(OP, np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8), iterations=self.dila_iter)
            roi = np.array(roi, dtype=np.float32)
            roi = roi.reshape( 1, self.imageHeight, self.imageWidth, 1) / 255 * 2 - 1

            OP[OP==255]=1
            GT = np.reshape(trainY, [self.imageHeight,self.imageWidth])
            GT[GT> 0]=1
            GT[GT<=0]=0
            areaOP = np.sum(OP)
            areaGT = np.sum(GT)
            accuracyDC = 1 - ( np.sum(np.abs(GT-OP)) / (areaOP + areaGT) )
            dataSet.append(readNow)
            accuracySetDC.append(accuracyDC*100)
            sumDiffDC += accuracyDC
            fileDC.write('Data: %04d Accuracy= %.9f \n'%(readNow,accuracyDC))

        fileAvgDC.write('%02d Average_Accuracy: %.9f \n'%(self.testData,(sumDiffDC/self.testSize)))
        fileDC.close()
        fileAvgDC.close()

        plt.bar(dataSet,accuracySetDC)
        plt.xlabel('data')
        plt.ylabel('accuracy(%)')
        plt.ylim(0,100)
        plt.savefig(self.outputDir + 'accuracyDC_%04d.png'%(self.testData))
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