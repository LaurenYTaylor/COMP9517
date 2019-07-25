import cv2
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from tqdm import tqdm


kernel_size = 5


def make_list():
    input_files = []
    label_files = []
    for root, dirs, file in os.walk('./images'):
        input_files.append(file)
    for root, dirs, file in os.walk('./labels'):
        label_files.append(file)
    final_files = []
    i = 0
    while i < len(input_files[0]):
        final_files.append([input_files[0][i], label_files[0][i]])
        i += 1
    return final_files


def prcoess_image(image, image_label):
    i = 0
    #sift = cv.xfeatures2d.SIFT_create()
    shape = image.shape
    inputs = []
    labels = []
    while i < (shape[0] - 10):
        j = 0
        while j < (shape[1] - 10):
            #patch = image[i:(i+kernel_size*2), j:(j+kernel_size*2)]
            #patch_label = image_label[i:(i+kernel_size*2), j:(j+kernel_size*2)]
            center_pixel = image[(i+kernel_size), (j+kernel_size)]
            center_pixel_label = image_label[i][j]

            # Can find SIFT features of patch here but i think this will take very long as it has to do this for every pixel!
            #kp, des = sift.detectAndCompute(patch, None)
            #kp_labels, des_labls = sift.detectAndCompute(patch_label, None)

            inputs.append(center_pixel)
            if center_pixel_label == 255:
                labels.append(1)
            elif center_pixel_label == 0:
                labels.append(0)
            else:
                print('ERROR IN LABEL PIXEL INTENSITY')
                print(center_pixel_label)
                exit()
            j += 1
        i += 1


    return inputs, labels

# This is for one image only! Can loop to obtain data inputs for all images
inputs = []
labels = []

filenames = make_list()

# Make list of image files
pbar = tqdm(total=len(filenames))
print('Processing images...')
for files in filenames:
    img_name = files[0]
    label_name = files[1]

    image = cv2.imread('./images/' + img_name)

    # Load the label
    image_label = cv2.imread('./labels/' + label_name)
    gray_label = cv2.cvtColor(image_label, cv2.COLOR_BGR2GRAY)


    # Perform Otsu thresholding on the image label to get binary labels
    ret, thresholded_label = cv2.threshold(gray_label, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # This is applying the blurring
    blurred = cv2.medianBlur(image, 5)

    # Now pad the image with ten pixels
    bordered = cv2.copyMakeBorder(blurred, kernel_size, kernel_size, kernel_size, kernel_size, cv2.BORDER_CONSTANT, value=0)

    # Now iterate through the image with a nxn mask and return something??
    inputs_image, labels_image = prcoess_image(bordered, thresholded_label)
    inputs += inputs_image
    labels += labels_image
    pbar.update(1)


inputs = np.array(inputs)
labels = np.array(labels)


# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    inputs, labels, test_size=0.2)


print("Training...")
'''

# Train k-NN model
classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train, y_train)

'''
classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, verbose=1)
classifier.fit(X_train, y_train)

# Take a new dataset and predict the label of each datapoint
y_pred = classifier.predict(X_test)

expected = y_test
predicted = classifier.predict(X_test)

print("Classification report %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

'''
cv2.imshow('bordered', processed_image)
cv2.waitKey(0)
'''




