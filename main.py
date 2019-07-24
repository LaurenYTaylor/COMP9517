import numpy as np
import cv2
import glob
import sys

image = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)

#img = cv2.GaussianBlur(image,(513, 513),0)
#output = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)
"""
Hypothetically the one that was most correct but resultent image lost too much detail
"""

#img = cv2.GaussianBlur(image,(17,17),0)
#output = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)
"""
A local minimum but displays how adaptive thresholding is unsuitable for this task
Noise removal requires significant blurring which by the time noise is removed, too much detail is lost
"""

#ree, output = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
"""
Any filtering reduced global thresholding performance
This is because it basically filtered everything to white
Bluring made it filter even more (if any black was remaining) to white
"""

#img = cv2.medianBlur(image, 555)
#ree, output = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
"""
Best numerical performance for otsu
Absolutely terrible actual performance as all detail is lost
"""

#img = cv2.GaussianBlur(image,(31, 31),0)
#ree, output = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
"""
First local minimum of otsu using gaussian filtering
Good performance for basic filtering but too much black
"""

img = cv2.medianBlur(image, 11)
ree, output = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
"""
First local minimum of otsu using median filtering
Best performance for basic filtering
"""

cv2.imshow('image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('image.jpg', output)
