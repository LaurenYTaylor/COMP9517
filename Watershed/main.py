import cv2
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


kernel_size = 5


def prcoess_image(image, image_label):
    i = 0
    sift = cv.xfeatures2d.SIFT_create()
    shape = image.shape
    while i < (shape[0] - 10):
        j = 0
        while j < (shape[1] - 10):
            patch = image[i:(i+kernel_size*2), j:(j+kernel_size*2)]
            patch_label = image_label[i:(i+kernel_size*2), j:(j+kernel_size*2)]
            center_pixel = image[(i+kernel_size), (j+kernel_size)]
            center_pixel_label = image_label[(i+kernel_size), (j+kernel_size)]

            # Can find SIFT features of patch here but i think this will take very long as it has to do this for every pixel!
            kp, des = sift.detectAndCompute(patch, None)
            kp_labels, des_labls = sift.detectAndCompute(patch_label, None)

            '''
            Do something here??? What is truth value of centre pixel? Manipulate the Kp of input and Kp label? 
            '''

            j += 1
        i += 1

# This is for one image only! Can loop to obtain data inputs for all images

img_name = 'train-volume00.jpg'
label_name = 'train-labels00.jpg'

image = cv2.imread('./images/' + img_name)

# Load the label
image_label = cv2.imread('./labels/' + label_name)

# This is applying the blurring
blurred = cv2.medianBlur(image, 5)

# Now pad the image with ten pixels
bordered = cv2.copyMakeBorder(blurred, kernel_size, kernel_size, kernel_size, kernel_size, cv2.BORDER_CONSTANT, value=0)

# Now iterate through the image with a nxn mask and return something??
#prcoess_image(bordered)

# Once we have the features and labels from the process function...
features, labels = False, False

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2)

# Train k-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

'''
OR use random forest:
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)
'''


# Take a new dataset and predict the label of each datapoint
y_pred = knn.predict(X_test)

expected = y_test
predicted = knn.predict(X_test)

print("Classification report %s:\n%s\n"
      % (knn, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

'''
cv2.imshow('bordered', processed_image)
cv2.waitKey(0)
'''




