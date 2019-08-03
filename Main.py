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
from keras.regularizers import l2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator



os.chdir("C:/Users/Ben/Desktop/Project")

file_name = glob.glob('C:/Users/Ben\Desktop/Project/data/images/*.jpg')
labels = glob.glob('C:/Users/Ben\Desktop/Project/data/labels/*.jpg')

labelsI = [int(i.split('-volume')[1][0:2]) for i in file_name]
labelsO = [int(i.split('-labels')[1][0:2]) for i in labels]
fileI = [i.split('\\')[1] for i in file_name]
fileO = [i.split('\\')[1] for i in labels]

Input = {'filename':fileI,'label':labelsI,'filedir':file_name}
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
    x = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if inputs.shape != num_filters:
        inputs = Conv2D(filters=num_filters,
                          kernel_size=(1, 1),
                          strides=(1,1),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(inputs)
    x = keras.layers.add([x, inputs])

    return x

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "images",mask_save_prefix  = "labels",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    label_datagen = ImageDataGenerator(**aug_dict)

    X = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    Y = label_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_gen = zip(X, Y)
    for (img,lab) in train_gen:
        img = img / 255
        lab = lab / 255
        lab[lab >= 0.5] = 1
        lab[lab < 0.5] = 0
        yield (img,lab)


def uresnet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = keras.layers.Input(input_size)
    resblock1 = resnet_block(inputs,64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(resblock1)
    resblock2 = resnet_block(pool1,128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(resblock2)
    resblock3 = resnet_block(pool2,256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(resblock3)
    resblock4 = resnet_block(pool3,512)
    drop4 = Dropout(0.5)(resblock4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    resblock5 = resnet_block(pool4,1024)
    drop5 = Dropout(0.5)(resblock5)
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    resblock6 = resnet_block(merge6,512)
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(resblock6))
    merge7 = concatenate([resblock3,up7], axis = 3)
    resblock7 = resnet_block(merge7,256)
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(resblock7))
    merge8 = concatenate([resblock2,up8], axis = 3)
    resblock8 = resnet_block(merge8,128)
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(resblock8))
    merge9 = concatenate([resblock1,up9], axis = 3)
    resblock9 = resnet_block(merge9,64)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(resblock9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    print(model.summary())

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

if __name__ == '__main__':
    data_gen_args = dict(rotation_range=90,\
                    width_shift_range=0.1,\
                    height_shift_range=0.1,\
                    shear_range=0.05,\
                    zoom_range=0.05,\
                    horizontal_flip=True,\
                    vertical_flip=True,\
                    fill_mode='wrap')
    myGene = trainGenerator(1,'data/','images','labels',data_gen_args,save_to_dir = 'data/Out/')
    model = uresnet()
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=2000,epochs=64,callbacks=[model_checkpoint])

# use BORDER_REFLECT_101
#samplewise_center = True,\
 #                   samplewise_std_normalization=True,\


    
