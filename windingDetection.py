# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:58:36 2019

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
    #    for winding_x,winding_y,winding_w,winding_h in windings:
    #    	cv2.rectangle(cimg, (winding_x, winding_y), (winding_x+winding_w, winding_y+winding_h), (0,255,0), 1)
        try:
            windings = np.uint16(windings*400/32)
            print(windings)
        except:
            windings = ()
    
        for winding_x,winding_y,winding_w,winding_h in windings:
#            winding_x = winding_x+190
#            winding_y = winding_y+370
            if winding_y > 190 and winding_y <240 and winding_x > 370 and winding_x < 420:
                cv2.rectangle(frame, (winding_x, winding_y), (winding_x+winding_w, winding_y+winding_h), (0,255,0), 2)
        
        cv2.imshow("detection",frame)
    #    cv2.imshow("detection",cimg)
        cv2.waitKey(1)
    except:
        cv2.destroyAllWindows()
        cap.release()
        break
'''
path = "g:/CV_Library/Winding_data/winding2/train/imgs/"
file_list = os.listdir(path)

shuflle = "/*.jpg"
for image_path in glob.glob(path+shuflle):
    img = cv2.imread(image_path)

    img=cv2.GaussianBlur(img,(11,11),5)
    row,col,_ = img.shape 
    rate = 32/400
    cimg = np.zeros((int(rate*row),int(rate*col),3))
    for s in range(3):
        cimg[:,:,s] = cv2.resize(img[:,:,s],(int(rate*col),int(rate*row)))
        cimg = np.uint8(cimg)
    cimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
    windings = winding_haar.detectMultiScale(cimg, 1.03, 5)
#    for winding_x,winding_y,winding_w,winding_h in windings:
#    	cv2.rectangle(cimg, (winding_x, winding_y), (winding_x+winding_w, winding_y+winding_h), (0,255,0), 1)
    try:
        windings = np.uint16(windings*400/32)
    except:
        windings = ()

    for winding_x,winding_y,winding_w,winding_h in windings:
    	cv2.rectangle(img, (winding_x, winding_y), (winding_x+winding_w, winding_y+winding_h), (0,255,0), 2)
    
    cv2.imshow("detection",img)
#    cv2.imshow("detection",cimg)
    cv2.waitKey(1)
    
#    show(cimg)

'''