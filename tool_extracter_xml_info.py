# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 08:56:34 2019

@author: Administrator
"""
import xml.etree.ElementTree as ET
classes = ["n00000001"]#因为我的数据集只有一个类别
def convert(size, box):#voc_label.py 自带的函数，没有修改
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
def convert_annotation(image_id):
    #in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    in_file = open('g:/CV_Library/Winding_data/winding3/train/xml/%s.xml'%(image_id))#与图片对应的xml文件所在的地址
    out_file = open('g:/CV_Library/Winding_data/winding3/train/xml/%s.txt'%(image_id),'w') #与此xml对应的转换后的txt，这个txt的保存完整路径
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')  #访问size标签的数据
    w = int(size.find('width').text)#读取size标签中宽度的数据
    h = int(size.find('height').text)#读取size标签中高度的数据

    for obj in root.iter('object'):
       # difficult = obj.find('difficult').text   #由于自己的文件里面没有diffcult这一个标签，所以就屏蔽之
        cls = obj.find('name').text
        if cls not in classes :#or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')   #访问boundbox标签的数据并进行处理，都按yolo自带的代码来，没有改动
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

#image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()  #之前代码是按照sets里面的字符来访问保存有图片名字的train或者val的txt文件
#image_ids = open('g:/CV_Library/Winding_data/winding3/train/train.txt').read().strip().split()  #如果是训练集数据打开这一行，注释下一行
image_ids = open('g:/CV_Library/Winding_data/winding3/train/val.txt').read().strip().split()  #如果是验证数据集数据打开这一行，注释上一行
#list_file = open('%s_%s.txt'%(year, image_set), 'w')
#list_file = open('infrared_train.txt', 'w')     #把结果写入到indrared_train.txt文件中，如果是训练集数据打开这一行，注释下一行
list_file = open('infrared_val.txt', 'w')     #把结果写入到indrared_train.txt文件中，如果是验证数据集数据打开这一行，注释上一行
for image_id in image_ids:
    #list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
    list_file.write('g:/CV_Library/Winding_data/winding3/train/imgs/%s.jpg\n'%(image_id))  #把每一用于训练或验证的图片的完整的路径写入到infrared_train.txt中  这个文件会被voc.data yolo.c调用
    convert_annotation(image_id)   #把图片的名称id传给函数，用于把此图片对应的xml中的数据转换成yolo要求的txt格式
list_file.close() #关闭文件
