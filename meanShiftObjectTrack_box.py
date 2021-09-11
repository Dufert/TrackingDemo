# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 09:50:52 2019

@author: Dufert
"""
import datetime as dt
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show(img):
    plt.figure(figsize=[7,7]),plt.imshow(img,cmap='gray'),plt.show()
    
#鼠标事件
def get_rect(im, title='get_rect'):
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):
        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                mouse_params['current_pos'], (255, 0, 0))

        cv2.imshow(title, im_draw)
        cv2.waitKey(10)

    cv2.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))

    return (tl, br)

cap = cv2.VideoCapture(r"d:\MyTestVideo_1.mp4")

# 读取摄像头第一帧图像
ret, frame = cap.read()
frame = cv2.GaussianBlur(frame,(3,3),2)
# 初始化位置窗口
a1,a2 = get_rect(frame, title='get_rect')
r,h,c,w = a1[1],a2[1]-a1[1],a1[0],a2[0]-a1[0]

box_a1,box_a2 = get_rect(frame, title='get_rect')
box_r,box_h,box_c,box_w = box_a1[1],box_a2[1]-box_a1[1],box_a1[0],box_a2[0]-box_a1[0]

cha_r = box_r - r
cha_c = box_c - c

track_window = (c,r,w,h)

# 设置所要跟踪的ROI
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
show(hsv_roi)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# 设置终止条件
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    start = dt.datetime.now()
    ret ,frame = cap.read()
    frame = cv2.GaussianBlur(frame,(3,3),2)
    try:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        cv2.rectangle(frame, (x+cha_c,y+cha_r), (x+cha_c+box_w,y+cha_r+box_h), (0,0,255),2)
        cv2.imshow('img2',frame)
        k = cv2.waitKey(1)
        end = dt.datetime.now()
        print(end - start)
    except:
        cv2.destroyAllWindows()
        cap.release()
        break
    
