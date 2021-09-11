# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:50:52 2019
对视频数据的遮挡检测、目标跟踪、卷扬识别的一系列功能实现
@author: Dufert
"""
import cv2
import warnings
import datetime
import numpy as np

warnings.filterwarnings('ignore')
class track_occlusion_class:
    
    def __init__(self):
        
        self.__occlution_count = 0
        self.bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)

#        r,h,c,w =226,55,324,26
        frame = cv2.imread("./img00001.jpg")
        r,h,c,w = 245, 330, 430, 340

        self.box = (r,h,c,w)
        box_r,self.box_h,box_c,self.box_w = 245, 330, 430, 340
        
        self.cha_r = box_r - r
        self.cha_c = box_c - c
        
        #KCF tracker初始化
        self.tracker = cv2.TrackerMOSSE_create()
#        self.tracker = cv2.TrackerTLD_create()
        self.tracker.init(frame, self.box)
        
        #Meanshift tracker初始化
        self.track_window = (c,r,w,h)

        # 设置所要跟踪的ROI
        roi = frame[r:r+h, c:c+w]
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)
        
        # 设置终止条件
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        


    def tracking_adjust(self,frames):
        start = datetime.datetime.now()
        ok, self.box = self.tracker.update(frames)
        bbox = self.box
        if ok and ((bbox[2] - 430)**2 + (bbox[0] - 245)**2)**0.5 < 20:
            pass
        else:
            frame = cv2.imread("./img00001.jpg")
            r,h,c,w = 245, 330, 430, 340
    
            self.box = (r,h,c,w)
            box_r,self.box_h,box_c,self.box_w = 245, 330, 430, 340
            
            self.cha_r = box_r - r
            self.cha_c = box_c - c
            
            self.tracker = cv2.TrackerMOSSE_create()
            self.tracker.init(frame, self.box)
            ok, self.box = self.tracker.update(frames)
            bbox = self.box
            if ~ok:
                bbox = (245, 330, 430, 340)
                print("第二次跟踪失败，选择初始框定输出")
#        if ((bbox[2] - 324)**2 + (bbox[0] - 226)**2)**0.5 >20:
#            bbox = (226,55,324,26)
        print(bbox)
        p1 = (int(bbox[2]), int(bbox[0]))
        p2 = (int(bbox[2] + bbox[3]), int(bbox[0] + bbox[1]))
        cv2.rectangle(frames, p1, p2, (255,0,0), 2, 1)
        
        x,y,z,h = (int(bbox[2] + self.cha_c), int(bbox[0] + self.cha_r),
                   int(bbox[2] + self.cha_c + self.box_w), int(bbox[0] + self.cha_r + self.box_h))
        
        cv2.rectangle(frames, (x,y), (z,h), (0,255,255),2,1)
        window = (y,h,x,z)
        end = datetime.datetime.now()
        
        elapsed_time = (end - start).total_seconds()
        FPS = 1 / elapsed_time
        
        print_log = 'FPS:  %.3f , track elapsed time: %.3f'%(FPS,elapsed_time)
        
        return window,frames,print_log
    
    
    def tracking_meanshift(self,frames):
        start = datetime.datetime.now()
        hsv = cv2.cvtColor(frames, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)
        flag = 0
        # apply meanshift to get the new location
        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        
        if ((self.track_window[0] - 324)**2 + (self.track_window[1] - 226)**2)**0.5 >30:
            self.track_window = (323, 227, 26, 55)
            flag = 1
        # Draw it on image
        x,y,w,h = self.track_window
        window = (y+self.cha_r,y+self.cha_r+self.box_h,
                  x+self.cha_c,x+self.cha_c+self.box_w)
        
        cv2.rectangle(frames, (x,y), (x+w,y+h), (0,255,0),2)
        cv2.rectangle(frames, (x+self.cha_c,y+self.cha_r), 
                      (x+self.cha_c+self.box_w,y+self.cha_r+self.box_h), 
                      (0,0,255),2)

        end = datetime.datetime.now()
        
        elapsed_time = (end - start).total_seconds()
        FPS = 1 / elapsed_time
        if flag == 0:
            print_log = 'FPS:  %.3f , track elapsed time: %.3f'%(FPS,elapsed_time)
        else:
            print_log = "adjust position"
        
        return window,frames,print_log
    

    def occlusion_detection(self,frame,window=()):
        start = datetime.datetime.now()
        frames = frame[window[0]:window[1], window[2]:window[3]]
        frames = cv2.GaussianBlur(frames,(3,3),2)
        fgmask = self.bs.apply(frames) # 背景分割器，该函数计算了前景掩码
        
        th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th,np.ones((2,3)))
        
        self.__occlution_count += 1
        binary_sum = np.sum(th)/255
        if self.__occlution_count > 8 and binary_sum > 4000:
            occ_det_result = 2
        else:
            occ_det_result = 0
        end = datetime.datetime.now()
        
        FPS = 1/(end - start).total_seconds()
        
        print_log = "FPS: %.3f , Occlusion Result: %d , value %d"%(FPS,occ_det_result,binary_sum)  
        
        return occ_det_result,print_log
    
    
