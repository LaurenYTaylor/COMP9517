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

os.chdir("C:/Users/bsmit/OneDrive/Desktop/Comp9517/project/")

file_name = glob.glob('C:/Users/bsmit/OneDrive/Desktop/Comp9517/project/data/images/*.jpg')
labels = glob.glob('C:/Users/bsmit/OneDrive/Desktop/Comp9517/project/data/labels/*.jpg')

labelsI = [int(i.split('-volume')[1][0:2]) for i in file_name]
labelsO = [int(i.split('-labels')[1][0:2]) for i in labels]
fileI = [i.split('\\')[1] for i in file_name]
fileO = [i.split('\\')[1] for i in labels]

Input = {'filename':fileI,'label':labelsI}
Output = {'filename':fileO,'label':labelsO}

Input = pd.DataFrame(data=Input)
Output = pd.DataFrame(data=Output)



def DataRot(image,rotation = 0):
    hieght, width = image.shape
    M = cv.getRotationMatrix2D((width/2,hieght/2),rotation,1)
    out = cv.warpAffine(image,M,(width,hieght))
    return out

def DataFlip(image, flip = 0):
    out = cv.flip(image, flip)
    return out



    
