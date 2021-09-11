# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:30:21 2019

@author: Administrator
"""
import cv2
import numpy as np


ocsvm = cv2.ml.SVM_create()
ocsvm.setType(cv2.ml.SVM_ONE_CLASS)
ocsvm.setKernel(cv2.ml.SVM_RBF)

ocsvm.setGamma(0.04)
ocsvm.setNu(0.001)

train_label = np.ones((len(train_label),),np.int32)

ocsvm.train(train_data, cv2.ml.ROW_SAMPLE,train_label)
_,result = ocsvm.predict(train_data)
result = np.int32(result.flatten())
support_vector_num,_ = np.shape(ocsvm.getSupportVectors())
print(support_vector_num)
print("error：%.4f"%(np.sum(result^train_label)/len(train_label)))

cv_labels = np.int32((cv_label+1)/2)
_,cvresult = ocsvm.predict(cv_data)
cvresult = np.int32(cvresult.flatten())
print("error：%.4f"%(np.sum(cvresult^cv_labels)/len(cv_labels)))

