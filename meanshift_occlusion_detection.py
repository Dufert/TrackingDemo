# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:31:23 2019

@author: Administrator
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img):
    plt.figure(figsize=[7,7]),plt.imshow(img,cmap='gray'),plt.show()

r,h,c,w = 245, 330, 430, 340
track_window = (c,r,w,h)

frame = cv2.imread(r"g:\CV_Library\Winding_data\adjust_para_data\imgs\img00015.jpg")
gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

gray_img = gray_img[245:575,430:770]
clache_1 = cv2.createCLAHE(clipLimit=32,tileGridSize=(16,16))
import datetime 
start = datetime.datetime.now()
adap_img = clache_1.apply(gray_img)
adap_img = cv2.medianBlur(np.uint8(adap_img),3)
end = datetime.datetime.now()

print((end - start).total_seconds())
show(adap_img)

import skimage.feature as ft

hog = ft.hog(adap_img)



def show(img):
    plt.figure(figsize=[7,7]),plt.imshow(img,cmap='gray'),plt.show()
# 设置初始化的窗口位置
r,h,c,w = 245, 330, 430, 340
track_window = (c,r,w,h)

cap = cv2.VideoCapture(r"k:\DATA\testWithOcclusion.mp4")
ret, frame= cap.read()
#frame = cv2.GaussianBlur(frame,(3,3),5)
roi = frame[r:r+h, c:c+w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
roi_hist = cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


enter_ascii = 13
cv2.namedWindow("test",0)

frame_num = cap.get(7)
frame_round = frame_num / 100 -1

break_flag = True
for i in range(int(frame_round)):
    if break_flag == False:
        break
    a = np.random.rand(1)*frame_round
    print("随机起始帧：%d"%np.int(np.floor(a*100)))
    cap.set(cv2.CAP_PROP_POS_FRAMES,np.int(np.floor(a*100)))
    count = 100
    while count>0:
        _, frame = cap.read()
#        frame = cv2.GaussianBlur(frame,(3,3),5)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
        # 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
        ret, window = cv2.meanShift(dst, track_window, term_crit)
        x,y,w,h = window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow("test",frame)
        if ret > 4:
            print('存在遮挡')
            print(ret)
        k = cv2.waitKey(1)
        if k == enter_ascii:
            break_flag = False
            break
        elif k == ord('q'):
            break
        count -= 1
cv2.destroyAllWindows()
