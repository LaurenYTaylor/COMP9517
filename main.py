import cv2
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import glob
import sys
import random

# SiftDetector from lab02
class SiftDetector():
	def __init__(self, norm="L2", params=None):
		self.detector=self.get_detector(params)
		self.norm=norm

	def get_detector(self, params):
		if params is None:
			params={}
			params["n_features"]=0
			params["n_octave_layers"]=3
			params["contrast_threshold"]=0.04
			params["edge_threshold"]=10
			params["sigma"]=1.6

		detector = cv2.xfeatures2d.SIFT_create(
				nfeatures=params["n_features"],
				nOctaveLayers=params["n_octave_layers"],
				contrastThreshold=params["contrast_threshold"],
				edgeThreshold=params["edge_threshold"],
				sigma=params["sigma"])

		return detector

def getSift(sample):
	# extract features
	kp = sift.detect(sample, None)
	# extract feature data
	vectors = []
	#Take first 5 features
	iterate = 5 if len(kp) > 5 else len(kp)
	zeros = numFeat-iterate
	l = 0
	while(l < zeros):
		vectors += [0,0,0,0,0,0,0]
		l += 1
	for k in range(iterate):
		vectors += [kp[k].angle,kp[k].class_id, kp[k].octave, kp[k].pt[0], kp[k].pt[1], kp[k].response, kp[k].size]
	return vectors


# seed
random.seed(4)

# sample size (actually double +1 of value set here)
sampleSize = 10

# feature count
numFeat = 5

# Initialise lists
features = []
truths = []

# get filenames
images = glob.glob("data/images/*.jpg")
labels = glob.glob("data/labels/*.jpg")

# initialise sift
sift_a = SiftDetector()
params={}
params["n_features"]=5
params["n_octave_layers"]=3
params["contrast_threshold"]=0.04
params["edge_threshold"]=10
params["sigma"]=1.6

sift = sift_a.get_detector(params)

# for each file
for i in range(24):
	# open files
	image = cv2.cvtColor(cv2.imread(images[i]), cv2.COLOR_BGR2GRAY)
	label = cv2.cvtColor(cv2.imread(labels[i]), cv2.COLOR_BGR2GRAY)
	# iterate 100 times
	for j in range(200):
		# generate random centre
		centrey = random.randrange(sampleSize, image.shape[0]-sampleSize)
		centrex = random.randrange(sampleSize, image.shape[1]-sampleSize)
		# get truth of centre
		truth = 1 if label[centrey][centrex] < 128 else 0
		# extract subimage
		sample = image[centrey-sampleSize:centrey+sampleSize, centrex-sampleSize:centrex+sampleSize]
		
		vectors = getSift(sample)

		features.append(vectors)
		truths.append(truth)

# Train knn with data
knn = KNeighborsClassifier(n_neighbors=101)
knn.fit(features, truths)

for i in range(24, 25):
	image = cv2.cvtColor(cv2.imread(images[i]), cv2.COLOR_BGR2GRAY)
	padImage = cv2.copyMakeBorder(image, sampleSize, sampleSize, sampleSize, sampleSize, cv2.BORDER_REFLECT)
	newImage = np.zeros((image.shape[0],image.shape[1]), np.uint8)
	for j in range(sampleSize, sampleSize+image.shape[0]):
		for k in range(sampleSize, sampleSize+image.shape[1]):
			sample = padImage[j-sampleSize:j+sampleSize, k-sampleSize:k+sampleSize]
			vectors = getSift(sample)
			newImage[j-sampleSize][k-sampleSize] = knn.predict_proba([vectors])[0][1]*255
	cv2.imshow("image", newImage)
	cv2.waitKey(0)

