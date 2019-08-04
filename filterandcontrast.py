import cv2
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MeanShift
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import glob
import sys
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tifffile import imsave


images = glob.glob("2*classout.jpg")

for k in range(len(images)):
	image = cv2.cvtColor(cv2.imread(images[k]), cv2.COLOR_BGR2GRAY)
	image = cv2.medianBlur(image, 11)
	max = 0
	min = 255
	# get min/max
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			if image[i][j] > max:
				max = image[i][j]
			if image[i][j] < min:
				min = image[i][j]
	# contrast stretch
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			image[i][j] = ((image[i][j] - min) / (max - min))*255.
	#convert to probability tiff
	imageout = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			imageout[i][j] = image[i][j]/255
	imsave("output"+str(k+24)+".tiff", imageout)