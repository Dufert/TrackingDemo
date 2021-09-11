# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:07:27 2019

@author: Dufert
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img):
    plt.figure(figsize=[7,7]),plt.imshow(img,cmap='gray'),plt.show()


winding_haar = cv2.CascadeClassifier(r"d:\winding\data_ga1\cascade.xml")
cv2.namedWindow("detection",0)

cap = cv2.VideoCapture(r"d:\MyTestVideo_1.mp4")
while cap.isOpened():
    try:
        flag,frame = cap.read()
        frame = cv2.GaussianBlur(frame,(11,11),5)
#        frame = frame[190:640,370:820]
        row,col,_ = frame.shape 
        rate = 32/400
        cimg = np.zeros((int(rate*row),int(rate*col),3))
        for s in range(3):
            cimg[:,:,s] = cv2.resize(frame[:,:,s],(int(rate*col),int(rate*row)))
            cimg = np.uint8(cimg)
        cimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
        windings = winding_haar.detectMultiScale(cimg, 1.03, 2)
        if windings != ():
            cv2.destroyAllWindows()
            break
    except:
        cv2.destroyAllWindows()
        break

windings = np.uint16(windings/rate)
for r,h,c,w in windings:
    pass
track_window = (c,r,w,h)

cap = cv2.VideoCapture(r"d:\MyTestVideo_1.mp4")

ret, frame= cap.read()
frame = cv2.GaussianBlur(frame,(3,3),5)
# 设置追踪的区域
roi = frame[r:r+h, c:c+w]


gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
filter_img = cv2.filter2D(gray_img,-1,kernel)
show(filter_img)
#ret,thresh_img = cv2.threshold(gray_img,113,255,cv2.THRESH_BINARY)

thresh_img = cv2.filter2D(gray_img,-1,kernel.transpose())

show(thresh_img)

# roi区域的hsv图像
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# 取值hsv值在(0,60,32)到(180,255,255)之间的部分

mask = cv2.inRange(hsv_roi, np.array((0., 32.,32.)), np.array((180.,255.,255.)))
# 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围

roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
# 归一化

roi_hist = cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# 设置终止条件，迭代10次或者至少移动1次
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame,(3,3),5)
    if ret == True:
        # 计算每一帧的hsv图像
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 计算反向投影
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 



