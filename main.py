import numpy as np
import cv2
import glob
import sys

image = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)

#img = cv2.GaussianBlur(image,(17,17),0)
#output = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)

img = cv2.medianBlur(image, 11)
ree, output = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#img = cv2.GaussianBlur(image,(31, 31),0)
#ree, output = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#ree, output = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

cv2.imshow('image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('image.jpg', output)
