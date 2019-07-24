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
	img = cv2.GaussianBlur(image[i],(blursize, blursize),0)
	#img = cv2.medianBlur(image[i], blursize)
	ree, output[i] = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

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


"""
Running this with various parameters of gaussian blur gave an average of
57561.53333333333 pixel difference with no blur
57574.96666666667 pixel difference with gaussian blur of square 3x3
57575.0 pixel difference with gaussian blue of square 5x5
Subsequent runs with larger squares repeated the last value

Running this with various parameters of median blur gave an average of
57561.53333333333 pixel difference with no blur
57574.5 pixel difference with median blur of parameter 3
57574.9 pixel difference with median blur of parameter 5
57574.933333333334 pixel difference with median blur of parameter 7
57575.0 pixel difference with median blue of paramter 9
Subsequent runs with larger parameters repeated the last value

Clearly, in regards to global thresholding, filtering the image has a slight negative impact
The best reult is obtained without any filtering and just performing global thresholding
That said, the results are quite bad with errors in around a quarter of the pixels
"""