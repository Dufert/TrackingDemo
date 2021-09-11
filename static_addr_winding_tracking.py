# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:50:52 2019

@author: Dufert
"""
import datetime as dt
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def show(img):
    plt.figure(figsize=[7,7]),plt.imshow(img,cmap='gray'),plt.show()
    
def position_read(path):
    posi = []
    coordinate_list = open(path+'positionMaxTrain.txt')
    position = coordinate_list.read()
    coordinate_list.close()
    position= position.split()
    for i in range(len(position)):
        posi.append(position[i].strip().split(','))
    posi = np.array(posi,np.float32)
    
    return posi

frame= cv2.imread(r"./img00001.jpg")

r,h,c,w =226,55,324,26
bbox = (r,h,c,w)

box_r,box_h,box_c,box_w = 250, 330, 430, 340

cha_r = box_r - r
cha_c = box_c - c
tracker = cv2.TrackerKCF_create()
ok = tracker.init(frame, bbox)

path = 'k:/train/'

file_list = os.listdir(path+'imgs/')
posi = position_read(path)



count = 0
for image_name in file_list:
    
    start = dt.datetime.now()
    frame = cv2.imread(path+'imgs/'+image_name)
    
    ok, bbox = tracker.update(frame)
    p1 = (int(bbox[2]), int(bbox[0]))
    p2 = (int(bbox[2] + bbox[3]), int(bbox[0] + bbox[1]))
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    
    x,y,z,h = (bbox[2]+cha_c,bbox[0]+cha_r, bbox[2]+cha_c+box_w,bbox[0]+cha_r+box_h)
    
    initial_coordinate = [x-24.75,y-21.05]
    dis = initial_coordinate - posi[count][0:2]
    print('distance: %.4f'%np.sum(dis**2)**0.5)
    count += 1
    
    end = dt.datetime.now()
    print((end - start).total_seconds())
    
    cv2.rectangle(frame, (int(x),int(y)), (int(z),int(h)), (0,0,255),2,1)
    cv2.imshow('img2',frame)
    k = cv2.waitKey(1)
    if k == 13:
        cv2.destroyAllWindows()
        break
