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


image = cv2.cvtColor(cv2.imread("probabilityIntensitySiftForest.jpg"), cv2.COLOR_BGR2GRAY)
max = 0
min = 255
for i in range(image.shape[0]):
	for j in range(image.shape[1]):
		if image[i][j] > max:
			max = image[i][j]
		if image[i][j] < min:
			min = image[i][j]

for i in range(image.shape[0]):
	for j in range(image.shape[1]):
		image[i][j] = ((image[i][j] - min) / (max - min))*255.
		image[i][j] = 255-image[i][j]

cv2.imshow("image", image)
cv2.waitKey(0)

# Perform watershed on knn probabilities
distance = ndi.distance_transform_edt(image)
maxes = peak_local_max(distance, indices = False, labels = image)
markers = ndi.label(maxes)[0]
ws_labels = watershed(-distance, markers, mask=image)



cv2.imshow("image", ws_labels)
cv2.waitKey(0)



