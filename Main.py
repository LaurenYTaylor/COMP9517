# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:24:05 2019

@author: bsmit
"""

"""
Data Augmentation
Check List
Rotating
Mirroring sampling
shifting potential
"""

import numpy as np
import cv2 as cv
import os
import glob
import pandas as pd
import keras
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.regularizers import l2
import os



# os.chdir("C:/Users/bsmit/OneDrive/Desktop/Comp9517/project/")

filenames = glob.glob('./data/images/*.jpg')
labels = glob.glob('./data/labels/*.jpg')

labelsI = [int(i.split('-volume')[1][0:2]) for i in filenames]
labelsO = [int(i.split('-labels')[1][0:2]) for i in labels]

fileI = [os.path.split(i)[1] for i in filenames]
fileO = [os.path.split(i)[1] for i in labels]

Input = {'filename':fileI,'label':labelsI,'filedir':filenames}
Output = {'filename':fileO,'label':labelsO,'filedir':labels}

Input = pd.DataFrame(data=Input)
Output = pd.DataFrame(data=Output)

def equalize(img):
    """Takes and image and returns the same image with the histogram equalized"""
    return cv.equalizeHist(img)

def dataRot(image,rotation = 0):
    hieght, width = image.shape
    M = cv.getRotationMatrix2D((width/2,hieght/2),rotation,1)
    out = cv.warpAffine(image,M,(width,hieght))
    return out

def dataFlip(image, flip = 0):
    out = cv.flip(image, flip)
    return out

def mirrorPadding(image,padding=0):
    out = cv.copyMakeBorder(image, padding, padding, padding, padding, cv.BORDER_REFLECT_101, None)
    return out

def resnet_block(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu'):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    # Two Conv layer block
    x = inputs
    x = conv(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = conv(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = keras.layers.add([x, inputs])

    return x

if __name__ == '__main__':
    img = cv.imread(Input.iloc[0,2],0)
    new_img = mirrorPadding(img,1)
    new_img = equalize(img)
    cv.imshow('new_img', new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# use BORDER_REFLECT_101
