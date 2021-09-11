# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:19:02 2019

@author: Administrator
"""
import cv2
import os,glob
import numpy as np
import matplotlib.pyplot as plt


def show(img):
    plt.figure(figsize=[3,3]),plt.imshow(img,cmap='gray'),plt.show()

#坐标 226 407
#坐标 603 785
#path = "g:/CV_Library/Winding_data/winding1/train/imgs/"
#path = "g:/CV_Library/Winding_data/winding4/CV/imgs/"
path = "d:/pos_old/"
file_list = os.listdir(path)

shuflle = "/*.jpg"
i = 0

for image_path in glob.glob(path+shuflle):
    cimg = np.zeros((32,32,3))
    img = cv2.imread(image_path)
#    cut_img = img
    img=cv2.GaussianBlur(img,(11,11),5)
#    cut_img = img[406+i:784+i,407:785]
    cha = i%5
    j = i%3
    if cha == 0:
        a = b = c = d = 0
    elif cha == 1:
        a = j;b = c = d = 0
    elif cha == 2:
        b = j;a = c = d = 0
    elif cha == 3:
        c = j;b = a = d = 0
    else:
        d = j;b = c = a = 0
#    a = -2;b = 2;c = -2;d = 2
    cut_img = img[215+i+10:615+i+10,396-i-10:796-i-10]
    for s in range(3):
        cimg[:,:,s] = cv2.resize(cut_img[:,:,s],(32,32))
    cimg = np.uint8(cimg)
    cimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
#    cv2.imwrite('i:/Anaconda/Anaconda_exercise/yolo/negative_gaussian_32/'+file_list[i],cimg)
#    cv2.imwrite('i:/Anaconda/Anaconda_exercise/yolo/positive_gaussian_32/'+file_list[i],cimg)
    cv2.imwrite('i:/Anaconda/Anaconda_exercise/yolo/neg_1/e'+str(i)+".jpg",cimg)
    i += 1
#    if i>25:
#        break
show(cut_img)

