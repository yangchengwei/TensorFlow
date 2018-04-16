
import os
import cv2
#import sys
import pygubu
import numpy as np
import tkinter as tk
from tkinter import filedialog
import base64
from icon import img

import tensorflow.contrib
import pygubu.builder.ttkstdwidgets

import tensorflow as tf
#model_path = './../version6/'
#sys.path.append(model_path)
from model import U_Net

class Application:
    def __init__(self, master):
        self.master = master

        #create builder
        self.builder = builder = pygubu.Builder()
        #load ui file
        builder.add_from_file('MOST_Project_UI.ui')
        #create a widget
        self.mainwindow = builder.get_object('window', master)
        #connect callback
        builder.connect_callbacks(self)

        #initial model
        self.sess = tf.Session()
#        self.model = U_Net(sess = self.sess, restoreDir = model_path + 'iter8_x2/')
        self.model = U_Net(sess = self.sess)
        self.model.build_model()
        self.model.restore('model46.ckpt')
        self.dila_iter = 8

        #initial item
        self.Spinbox_Interval = builder.get_object('Spinbox_Interval', master)
        self.Checkbutton_Show_Contour = builder.get_object('Checkbutton_Show_Contour', master)
        self.Checkbutton_Show_Contour.state(['!alternate']) # (alternate,) -> ()
        self.Checkbutton_Show_Contour.state(['selected']) # () -> (selected,)
        self.Checkbutton_Fine_Tune = builder.get_object('Checkbutton_Fine_Tune', master)
        self.Checkbutton_Fine_Tune.state(['!alternate']) # (alternate,) -> ()
        return
####################################################################################################
    def Button_Load_File_Click(self):
#        print('Button_Load_File_Click')
        file_name = filedialog.askopenfilename()
        if file_name == '':
            return
        self.input_file_name = file_name
        print ('Input_file:', file_name)
        
        self.frames_all = []
        self.contour_all = []
        self.now_frame = 0
        self.now_Interval = 10
        self.ix = 0
        self.iy = 0
        self.L_button_down = False
        self.R_button_down = False
        self.fine_tune = False
        self.Checkbutton_Fine_Tune.state(['!selected'])
        
        self.Spinbox_Interval.set(10)

        video = cv2.VideoCapture(self.input_file_name)
        success, frame = video.read()
        while(success):
            self.frames_all.append(np.array(frame, dtype=np.uint8))
            self.contour_all.append(np.zeros_like(frame[:,:,0], dtype=np.uint8))
            success, frame = video.read()
        
        cv2.destroyAllWindows()
        cv2.namedWindow(self.input_file_name)
        self.update_image()
        cv2.createTrackbar('frame', self.input_file_name, 0, len(self.frames_all)//self.now_Interval, self.Trackbar_Change)
        cv2.setMouseCallback(self.input_file_name, self.draw_contour)
        return
####################################################################################################
    def Button_Load_Init_Click(self):
        print('Button_Load_Init_Click')
        filename = filedialog.askopenfilename()
        if filename == '':
            return
        print ('Input_file:', filename)
        inputim = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        inputim[inputim>=128]=255
        inputim[inputim<128]=0
        self.contour_all[self.now_frame][0:360,0:800] = inputim
        self.update_image()
        return
####################################################################################################
    def Button_Save_Result_Click(self):
        print('Button_Save_Result_Click')
        filename = self.input_file_name.split('/')[-1].split('.')[0]
        print(filename)
        path_name = filedialog.askdirectory()
        if path_name == '':
            return
        path_name = path_name + '/' + filename + '/'
        print ('Output_path:', path_name)
        L = len(self.contour_all)
        os.mkdir(path_name)
        file_number = 0
        for i in range(L):
            print(i)
            if np.sum(self.contour_all[i])!=0 :
                cv2.imwrite(path_name+str(file_number)+'.bmp', self.contour_all[i][0:360,0:800])
                file_number = file_number + 1
        return
####################################################################################################
    def predict(self, contour_number, frame_number):
        input_ultrasound = cv2.resize(self.frames_all[frame_number][0:360,0:800], dsize=(self.model.imageWidth,self.model.imageHeight))
        input_roi = cv2.resize(self.contour_all[contour_number][0:360,0:800], dsize=(self.model.imageWidth,self.model.imageHeight))
        input_roi = cv2.dilate(input_roi, np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8), iterations=self.dila_iter)
        input_image = np.zeros([1, self.model.imageHeight,self.model.imageWidth,self.model.inputChannel], dtype=np.float32)
        input_image[0,:,:,0] = input_ultrasound[:,:,0] / 255 * 2 - 1
        input_image[0,:,:,1] = input_roi[:,:] / 255 * 2 - 1
        predict = self.model.sess.run(self.model.PD, feed_dict={self.model.IP : input_image})
        input_roi = np.reshape(predict, [self.model.imageHeight,self.model.imageWidth])
        input_roi = cv2.resize(input_roi, dsize=(800,360))
        input_roi[input_roi >= 0] = 255
        input_roi[input_roi < 0] = 0
        self.contour_all[frame_number][0:360,0:800] = input_roi

#        input_ultrasound = cv2.resize(self.frames_all[frame_number][0:256,144:656], dsize=(self.model.imageWidth,self.model.imageHeight))
#        input_roi = cv2.resize(self.contour_all[contour_number][0:256,144:656], dsize=(self.model.imageWidth,self.model.imageHeight))
#        input_roi = cv2.dilate(input_roi, np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8), iterations=self.dila_iter)
#        input_image = np.zeros([1, self.model.imageHeight,self.model.imageWidth,self.model.inputChannel], dtype=np.float32)
#        input_image[0,:,:,0] = input_ultrasound[:,:,0] / 255 * 2 - 1
#        input_image[0,:,:,1] = input_roi[:,:] / 255 * 2 - 1
#        predict = self.model.sess.run(self.model.PD, feed_dict={self.model.IP : input_image})
#        input_roi = np.reshape(predict, [self.model.imageHeight,self.model.imageWidth])
#        input_roi[input_roi >= 0] = 255
#        input_roi[input_roi < 0] = 0
#        self.contour_all[frame_number][0:256,144:656] = input_roi
        return
####################################################################################################
    def Button_Tracking_All_Click(self):
        print('Button_Tracking_All_Click')
        self.predict(0, 0)
        for frame in range(self.now_Interval, len(self.frames_all), self.now_Interval):
            print(frame)
            self.predict(frame-self.now_Interval, frame)
        self.update_image()
        return
####################################################################################################
    def Button_Tracking_Forward_Click(self):
        print('Button_Tracking_Forward_Click')
        for frame in range(self.now_frame + self.now_Interval, len(self.frames_all), self.now_Interval):
            print(frame)
            self.predict(frame-self.now_Interval, frame)
        self.update_image()
        return
####################################################################################################
    def Button_Tracking_Backward_Click(self):
        print('Button_Tracking_Backward_Click')
#        for frame in range(len(self.frames_all)-1-(len(self.frames_all)-1)%self.now_Interval-self.now_Interval,-1,-self.now_Interval):
        for frame in range(self.now_frame - self.now_Interval, -1, -self.now_Interval):
            print(frame)
            self.predict(frame+self.now_Interval, frame)
        self.update_image()
        return
####################################################################################################
    def Button_Clear_Current_Click(self):
        print('Button_Clear_Current_Click')
        self.contour_all[self.now_frame] = np.zeros_like(self.contour_all[self.now_frame])
        self.update_image()
        return
####################################################################################################
    def Button_Clear_All_Click(self):
        print('Button_Clear_All_Click')
        L = len(self.contour_all)
        for i in range(L):
            self.contour_all[i] = np.zeros_like(self.contour_all[i])
        self.update_image()
        return
####################################################################################################
    def Spinbox_Interval_Change(self):
#        print('Spinbox_Interval_Change')
        self.now_frame = 0
        self.now_Interval = int(self.Spinbox_Interval.get())
        cv2.createTrackbar('frame', self.input_file_name, 0, len(self.frames_all)//self.now_Interval, self.Trackbar_Change)
        return
####################################################################################################
    def Checkbutton_Fine_Tune_Change(self):
#        print('Checkbutton_Fine_Tune_Change')
        self.fine_tune = True if 'selected' in self.Checkbutton_Fine_Tune.state() else False
        self.update_image()
        return
####################################################################################################
    def Checkbutton_Show_Contour_Change(self):
#        print('Checkbutton_Show_Contour_Change')
        self.update_image()
        return
####################################################################################################
    def Trackbar_Change(self, x):
        self.now_frame = x * self.now_Interval
        self.update_image()
        return
####################################################################################################
    def update_image(self):
#        print('update_image')
        self.image_show = np.array(self.frames_all[self.now_frame], dtype=np.int16)
        if 'selected' in self.Checkbutton_Show_Contour.state():
            if self.fine_tune:
                contour = self.contour_all[self.now_frame]
                self.image_show[contour>0] = self.image_show[contour>0]+[-64,+64,+64]
                self.image_show[self.image_show>255] = 255
                self.image_show[self.image_show<0] = 0
            else:
#                print('self.show_contour == True')
                contour = self.contour_all[self.now_frame]
                contour = cv2.erode(contour, np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8), iterations=2)
                contour = self.contour_all[self.now_frame]-contour
                self.image_show[contour>0] = [0,255,255]
        if self.fine_tune:
            self.image_show = np.array(self.image_show[0:360,0:800], dtype=np.uint8)
            self.image_show = cv2.resize(self.image_show,(1600,720), interpolation=cv2.INTER_NEAREST)
        else:
            self.image_show = np.array(self.image_show, dtype=np.uint8)
        cv2.imshow(self.input_file_name, self.image_show)
        return
####################################################################################################
    def draw_contour(self,event,x,y,flags,param):
#        print('draw_contour')
        if self.fine_tune:
            color = (0,0,255)
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(self.contour_all[self.now_frame], (x//2,y//2), 4, 255, -1)
                self.update_image()
                self.L_button_down = True
            elif event == cv2.EVENT_LBUTTONUP:
                self.L_button_down = False
            elif self.L_button_down:
                cv2.circle(self.contour_all[self.now_frame], (x//2,y//2), 4, 255, -1)
                self.update_image()
            elif event == cv2.EVENT_RBUTTONDOWN:
                color = (255,0,0)
                cv2.circle(self.contour_all[self.now_frame], (x//2,y//2), 4, 0, -1)
                self.update_image()
                self.R_button_down = True
            elif event == cv2.EVENT_RBUTTONUP:
                self.R_button_down = False
            elif self.R_button_down:
                color = (255,0,0)
                cv2.circle(self.contour_all[self.now_frame], (x//2,y//2), 4, 0, -1)
                self.update_image()
            image_show_temp = np.copy(self.image_show)
            cv2.circle(image_show_temp, (x,y), 8, color, -1)
            cv2.imshow(self.input_file_name, image_show_temp)
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
#                print('cv2.EVENT_LBUTTONDOWN')
                self.ix = x
                self.iy = y
                self.L_button_down = True
            elif event == cv2.EVENT_LBUTTONUP:
#                print('cv2.EVENT_LBUTTONUP')
                cv2.rectangle(self.contour_all[self.now_frame],(self.ix,self.iy),(x,y),255,-1)
                self.update_image()
                self.L_button_down = False
            elif self.L_button_down:
#                print('self.L_button_down')
                image_show_temp = np.copy(self.image_show)
                cv2.rectangle(image_show_temp,(self.ix,self.iy),(x,y),(0,0,255),1)
                cv2.imshow(self.input_file_name, image_show_temp)
            elif event == cv2.EVENT_RBUTTONDOWN:
#                print('cv2.EVENT_RBUTTONDOWN')
                self.ix = x
                self.iy = y
                self.R_button_down = True
            elif event == cv2.EVENT_RBUTTONUP:
#                print('cv2.EVENT_RBUTTONUP')
                cv2.rectangle(self.contour_all[self.now_frame],(self.ix,self.iy),(x,y),0,-1)
                self.update_image()
                self.R_button_down = False
            elif self.R_button_down:
#                print('self.R_button_down')
                image_show_temp = np.copy(self.image_show)
                cv2.rectangle(image_show_temp,(self.ix,self.iy),(x,y),(255,0,0),1)
                cv2.imshow(self.input_file_name, image_show_temp)
        return
####################################################################################################

####################################################################################################

####################################################################################################

####################################################################################################

root = tk.Tk()

tmp = open('tmp.ico','wb+')
tmp.write(base64.b64decode(img))
tmp.close()
root.iconbitmap('tmp.ico')
os.remove('tmp.ico')

app = Application(root)
root.mainloop()

cv2.destroyAllWindows()
app.frames_all = []
app.contour_all = []