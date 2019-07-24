import numpy as np
import cv2
import glob
import sys

blursize = int(sys.argv[1])

if blursize % 2 == 0:
	sys.exit(1)

image = [None]*30

filenames = glob.glob("data/images/*.jpg")
for i, filename in enumerate(filenames):
	image[i] = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

truth = [None]*30

filenames = glob.glob("data/labels/*.jpg")
for i, filename in enumerate(filenames):
	truth[i] = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)

output = [None]*30

for i in range(30):
	#img = cv2.GaussianBlur(image[i],(blursize, blursize),0)
	img = cv2.medianBlur(image[i], blursize)
	output[i] = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)

differences = [None]*30

for i in range(30):
	outputVector = output[i].reshape(262144)
	truthVector = truth[i].reshape(262144)
	differenceVector = (outputVector-truthVector)/255
	for j in range(262144):
		if differenceVector[j] >= 0.5:
			differenceVector[j] = 1
		else:
			differenceVector[j] = 0
	differences[i] = np.sum(differenceVector)

print(np.mean(differences))


