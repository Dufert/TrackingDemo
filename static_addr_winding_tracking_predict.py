# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:50:52 2019

@author: Dufert
"""
import datetime as dt
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import warnings
import datetime
from sklearn.externals import joblib
from multhread_predict_class import multhread_predict_class
from read_feature_class import read_feature_class
from collections import deque

def show(img):
    plt.figure(figsize=[7,7]),plt.imshow(img,cmap='gray'),plt.show()
R = threading.Lock()
dp = deque([0, 0, 0, 0, 0, 0, 0], maxlen=7)

warnings.filterwarnings('ignore')
cap = cv2.VideoCapture(r'e:\DATA\190110\00000000031000000_test.mp4')

read_data = read_feature_class(2,2)
predict_data = multhread_predict_class(2)
clf = joblib.load(r"h:\Wroking In Zoomlion\project\Winding_system_for_model_changde_save\_center_nu0.001gamma0.16F10.953195772521389_ocsvm_model.m")
pca1 = joblib.load(r"h:\Wroking In Zoomlion\project\Winding_system_for_model_changde_save\pca_model.m")
# 初始化 tracking 区域
frame = cv2.imread(r"h:\Wroking In Zoomlion\project\Winding_system_for_model_changde_model_new_gabor\img00001.jpg")
frame = cv2.GaussianBlur(frame,(3,3),2)
r,h,c,w =226,55,324,26
box_r,box_h,box_c,box_w = 250, 330, 430, 340

cha_r = box_r - r
cha_c = box_c - c

track_window = (c,r,w,h)

# 设置所要跟踪的ROI
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# 设置终止条件
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )



def predict(threadName):
    global frame
    global read_data
    global pca1
    global clf
    global dp
    while cap.isOpened():
        try:
            flag,frame = cap.read()
            start = datetime.datetime.now()
            output,_,_ = predict_data.chooseFeatureOutput(frame,read_data,clf,pca1)
            end = datetime.datetime.now()
            dp.appendleft(output)
            if np.sum(dp) > 4:
               print('乱绳')
            print("Predict Result: %d , 耗时： %.4f"%(output, (end - start).total_seconds()))
            time.sleep(0.001)
        except:
            print('error in thread 1')
            break



def tracking_adjust(threadName):
    global cap
    global track_window
    while cap.isOpened():
        start = dt.datetime.now()
        _,frame = cap.read()
        frame = cv2.GaussianBlur(frame,(3,3),2)
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # Draw it on image
            x,y,w,h = track_window
            print(track_window)
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
            # cv2.rectangle(frame, (x+cha_c,y+cha_r), (x+cha_c+box_w,y+cha_r+box_h), (0,0,255),2)
            # cv2.imshow('img2',frame)
            # k = cv2.waitKey(1)
            end = dt.datetime.now()
            print((end - start).total_seconds())
            time.sleep(0.001)
        except:
            # cv2.destroyAllWindows()
            # cap.release()
            print('error in thread 2')
            break
