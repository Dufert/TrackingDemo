# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:01:47 2019
预测跟踪遮挡检查调用例子
@author: Dufert
"""
import cv2
import numpy as np
from track_occlusion_class import track_occlusion_class as toc

test = toc()

cap = cv2.VideoCapture(r'k:\DATA\testWithOcclusion.mp4')

enter_ascii = 13
cv2.namedWindow("test",0)

frame_num = cap.get(7)
frame_round = frame_num / 100 -1

for i in range(int(frame_round)):
    a = np.random.rand(1)*frame_round
    print(np.int(np.floor(a*100)))
    cap.set(cv2.CAP_PROP_POS_FRAMES,np.int(np.floor(a*100)))
    count = 100
    while count>0:
        flag,frame = cap.read()
#        window,img,track_log = test.tracking_meanshift(frame)
        window,img,track_log = test.tracking_adjust(frame)
        output_occ,occ_log = test.occlusion_detection(frame,window)
        img[0:120,0:1280,:] = 0
        cv2.putText(img,track_log+" "+str(window), (100,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255),2)
        cv2.putText(img,occ_log, (100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
        
        cv2.imshow("test",img)
        k = cv2.waitKey(1)
        if k == enter_ascii:
            break
        count -= 1
cv2.destroyAllWindows()
        
        