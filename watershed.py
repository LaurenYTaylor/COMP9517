import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift
from sklearn.ensemble import RandomForestClassifier

# run random forest classifier on image
# get ouput probability map
# run watershed on probability map
# Need to get output in form of tree

# following code is from lab 4
"""
img = Image.open(img_path)
img.thumbnail(size)  # Convert the image to 100 x 100
# Convert the image to a numpy matrix
img_mat = np.array(img)[:, :, :3]

# Step 1 - Convert the image to gray scale
# and convert the image to a numpy matrix
img_array = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
# Step 2 - Calculate the distance transform
# Hint: use     ndi.distance_transform_edt(img_array)
distance = ndi.distance_transform_edt(img_array)
# Step 3 - Generate the watershed markers
# Hint: use the peak_local_max() function from the skimage.feature library
# to get the local maximum values and then convert them to markers
# using ndi.label() -- note the markers are the 0th output to this function
maxes = peak_local_max(distance, indices = False, labels=img_array)
markers = ndi.label(maxes)[0]

# Step 4 - Perform watershed and store the labels
# Hint: use the watershed() function from the skimage.morphology library
# with three inputs: -distance, markers and your image array as a mask
ws_labels = watershed(-distance, markers, mask=img_array)
"""