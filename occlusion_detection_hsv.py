# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:46:52 2019

@author: Dufert
"""
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt 

path = "g:/CV_Library/Winding_data/winding1/train/imgs/"
imglist = os.listdir(path)

#for image_name in imglist:
#    image_path = path+image_name
#    img = cv2.imread(image_path)
#    img = img[270:550, 450:740,:]
#    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#    h_img = hsv_img[:,:,0]
#    h_vecter = cv2.calcHist([h_img],[0],None,[128],[0,256])/255
#    
#    cv2.imshow("h_img",h_img)
#    cv2.waitKey(1)
##    plt.plot(h_vecter),plt.show()
#    
#cv2.destroyAllWindows()

cap = cv2.VideoCapture(r'd:\CV_Library\MyTestVideo_1.mp4')

while(cap.isOpened()):
    _,img = cap.read()
    img = img[290:550, 460:740,:]
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_img = hsv_img[:,:,0]
    
    h_vecter = cv2.calcHist([h_img],[0],None,[128],[0,256])/255
    _,thresh_img = cv2.threshold(h_img,36,255,cv2.THRESH_BINARY)
    th_sum = np.sum(thresh_img)/255
    print(th_sum)
#    if th_sum > 16000:
#        print("存在遮挡")
    cv2.putText(img, str(th_sum), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
    cv2.imshow("h_img",thresh_img)
    cv2.imshow("img",img)
    cv2.waitKey(1)
cv2.destroyAllWindows()
