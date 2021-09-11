# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:00:58 2019

@author: Dufert
"""

import os
import xml.etree.ElementTree as ET
import glob

def xml_to_txt(indir,outdir):

    os.chdir(indir)#进入当前目录
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')

    for i, file in enumerate(annotations):

        file_save = file.split('.')[0]+'.txt'
        file_txt=os.path.join(outdir,file_save)
        f_w = open(file_txt,'w')

        # actual parsing
        in_file = open(file)
        tree=ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):

                xmlbox = obj.find('bndbox')
                xn = xmlbox.find('xmin').text
                xx = xmlbox.find('xmax').text
                yn = xmlbox.find('ymin').text
                yx = xmlbox.find('ymax').text

                f_w.write(xn+' '+yn+' '+xx+' '+yx+' ')

indir='g:/CV_Library/Winding_data/winding3/train/xml/'   #xml目录
outdir='g:/CV_Library/Winding_data/winding3/train/txt/'  #txt目录

xml_to_txt(indir,outdir)
