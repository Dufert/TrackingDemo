# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:26:22 2019

@author: Dufert
"""

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
frame = cv2.GaussianBlur(frame,(3,3),5)

kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

# 初始化位置窗口
a1,a2 = get_rect(frame, title='get_rect')
r,h,c,w = a1[1],a2[1]-a1[1],a1[0],a2[0]-a1[0]
track_window = (c,r,w,h)

roi = frame[r:r+h, c:c+w]
gray_img = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
filter_img = cv2.filter2D(gray_img,-1,kernel)
show(filter_img)

row, col = filter_img.shape
begin_row = 215
begin_col = 396
end_row = 615
end_col = 796


while True:
    flag, frame = cap.read()
    try:
        frame = cv2.GaussianBlur(frame,(3,3),5)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.filter2D(gray,-1,kernel)
    
        sum_list = []
        for i in range(begin_row,end_row-row,1):
            for j in range(begin_col,end_col-col,1):
                sums = np.sum(np.abs(filter_img - gray[i:i+row,j:j+col]))/(row*col)
                sum_list.append((sums,i,j))
    
        lists = np.array(sum_list)[:,0].flatten()
        addr = np.where(np.min(lists) == lists)
        _,y,x = sum_list[int(addr[0])]
        img2 = cv2.rectangle(gray, (x,y), (x+row,y+col), 255,2)
    
        cv2.imshow("show",img2)
        cv2.waitKey(1)
    except:
        cap.release()
        cv2.destroyAllWindows()
        break





