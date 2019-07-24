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
	ree, output[i] = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
41426.0 pixel difference with no blur
41370.4 pixel difference with gaussian blu of square 3x3
41319.566666666666 pixel difference with gaussian blu of square 5x5
41194.8 pixel difference with gaussian blu of square 7x7
41138.7 pixel difference with gaussian blu of square 9x9
40980.1 pixel difference with gaussian blu of square 11x11
40859.833333333336 pixel difference with gaussian blu of square 13x13
40703.26666666667 pixel difference with gaussian blu of square 15x15
40502.0 pixel difference with gaussian blu of square 17x17
40360.0 pixel difference with gaussian blu of square 19x19
57575.0 pixel difference with gaussian blu of square 21x21
Subsequent runs with larger squares repeated the last value

Running this with various parameters of median blur gave an average of
41426.0 pixel difference with no blur
41477.46666666667 pixel difference with median blur of parameter 3
41508.76666666667 pixel difference with median blur of parameter 5
41352.166666666664 pixel difference with median blur of parameter 7
40966.5 pixel difference with median blur of parameter 9
40664.333333333336 pixel difference with median blur of parameter 11
40695.96666666667 pixel difference with median blur of parameter 13
40959.46666666667 pixel difference with median blur of parameter 15
41272.53333333333 pixel difference with median blur of parameter 17
41662.9 pixel difference with median blur of parameter 19
41929.566666666666
42193.6
42376.46666666667
42549.36666666667
42755.2
42845.03333333333
42936.066666666666
43010.13333333333
43073.36666666667
43068.9


57561.53333333333
57574.5 pixel difference with median blur of parameter 3
57574.9 pixel difference with median blur of parameter 5
57574.933333333334 pixel difference with median blur of parameter 7
57575.0 pixel difference with median blue of paramter 9
Subsequent runs with larger parameters repeated the last value

Clearly, in regards to global thresholding, filtering the image has a slight negative impact
The best reult is obtained without any filtering and just performing global thresholding
That said, the results are quite bad with errors in around a quarter of the pixels
"""