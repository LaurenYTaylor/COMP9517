import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
import glob
import cv2
import numpy as np
import Foveation as fov
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model, Sequential
from keras import optimizers
import pickle
from sklearn.metrics import mean_absolute_error
import re
from keras.wrappers.scikit_learn import KerasClassifier


#LABEL OF 0 IS NON_MEMBRANE, LABEL OF 1 IS MEMBRANE

R=16
BLUR_RADIUS=int(R/4)
IMAGE_WIDTH=512

image_filenames = glob.glob('data/images/*')
label_filenames = glob.glob('data/labels/*')

images_dict = {}
labels_dict = {}

print('Creating image dictionary............')
for i in range(len(image_filenames)):
	images_dict[int(image_filenames[i][-6:-4])]=cv2.cvtColor(cv2.imread(image_filenames[i]), cv2.COLOR_BGR2GRAY)

print('Creating label dictionary.............')
for i in range(len(label_filenames)):
	image = cv2.cvtColor(cv2.imread(label_filenames[i]), cv2.COLOR_BGR2GRAY)
	white_indices = np.where(image>=170)
	for j in range(len(white_indices[0])):
		str = '{}_{}_{}'.format(i, white_indices[0][j], white_indices[1][j])
		labels_dict[str]=0
	black_indices = np.where(image<170)
	for j in range(len(black_indices[0])):
		str = '{}_{}_{}'.format(i, black_indices[0][j],black_indices[1][j])
		labels_dict[str]=1
with open('labels.pickle', 'wb') as f:
	pickle.dump(labels_dict, f)
# with open('../labels.pickle', 'rb') as f:
	# labels_dict=pickle.load(f)

# black_pixels = 0
# white_pixels = 0
# for key, value in labels_dict.items():
	# if value==0:
		# white_pixels+=1
	# else:
		# black_pixels+=1
# print(black_pixels) # 1727250 black pixels
# print(white_pixels) # 6137070 white pixels

def create_windows(image, label, stepSize, R=R):
	padded_image = cv2.copyMakeBorder(image, R, R, R, R, borderType=cv2.BORDER_REFLECT_101)
	windows={}
	for x in range(R, padded_image.shape[0]-R, stepSize):
		for y in range(R, padded_image.shape[1]-R, stepSize):
			start_x=x-R
			start_y=y-R
			window=padded_image[x-R:x+R+1,y-R:y+R+1]
			str="{}_{}_{}".format(label, start_x, start_y)
			windows[str]=window
	return windows

foveated_windows={}
max_windows=30
step_size=int(np.floor(IMAGE_WIDTH/max_windows))
train_windows={}
test_windows={}
for i in range(len(images_dict.keys())):
	label = list(images_dict.keys())[i]
	image = list(images_dict.values())[i]
	print('Creating pixel windows for image {}...........'.format(i))
	if label<23:
		print('Train image: '+str(label))
		train_windows.update(create_windows(image, label, step_size, R))
	else:
		print('Test image: '+str(label))
		test_windows.update(create_windows(image, label, step_size, R))
	#print('Foveating windows for image {}..........'.format(i))
	# foveator = fov.Foveation(BLUR_RADIUS)
	# j=0
	# for key in windows.keys():
		# print(j)
		# foveated_windows[key]=(foveator.foveate(windows[key]))
		# j+=1

train_keys, train_values = zip(*train_windows.items())

y_train = [labels_dict[key] for key in train_keys]

normalised_values=[]
for image in train_values:
	normalised_values.append(image/np.amax(image))

normalised_values=np.array(normalised_values)
X_train = normalised_values.reshape(normalised_values.shape[0], normalised_values.shape[1], normalised_values.shape[2], 1)

#Define the neural network model


def model(filter1=16,kernel1=2,pool1=2,kernel2=2,dropout1=0.5,dropout2=0.5,lr=0.0007):
	model = Sequential()
	model.add(Conv2D(filters=filter1, kernel_size=kernel1, input_shape=(X_train.shape[1],X_train.shape[2],1), activation='relu',data_format='channels_last', padding = 'same'))
	model.add(MaxPooling2D(pool_size=pool1))
	model.add(Dropout(dropout1))
	model.add(Flatten())
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(dropout2))
	model.add(Dense(1, activation='sigmoid'))
	oad = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss='binary_crossentropy', optimizer=oad, metrics=['accuracy'])
	return model

filter1 = [16, 32]
kernel1 = [2, 4]
pool1 = [2, 4]
kernel2 = [2, 4]
dropout1 = [0.3, 0.5]
dropout2 = [0.3, 0.5]
learning_rate = [0.0003, 0.0007, 0.001]
batch_size = [16, 32, 64]

param_dict = {'filter1': filter1, 'kernel1':kernel1, 'kernel2':kernel2, 'pool1':pool1,
				'dropout1': dropout1, 'dropout2': dropout2, 'lr': learning_rate,
				'batch_size': batch_size}
				

gs = GridSearchCV(KerasClassifier(build_fn=model, epochs=20), param_grid=param_dict, 
					scoring='accuracy', n_jobs=1, cv=3)

grid_result = gs.fit(X_train, y_train)

with open('results_file.txt', 'w') as f:
	f.write(str(grid_result.cv_results_))
	f.write('\n\n\nBest Score:\n')
	f.write(str(grid_result.best_score_))
	f.write('\nBest Params:\n')
	f.write(str(grid_result.best_params_))


# reconstructed_image = np.ones(test_image.shape)
# reconstructed_image = 255*reconstructed_image
# i=0
# for test_key, test_value in test_windows.items():
	# if(i%1000==0):
		# print(i)
	# match = re.match('([^_]*)_([^_]*)_([0-9]*)', test_key)
	# image_num = int(match.group(1))
	# pixel_x = int(match.group(2))
	# pixel_y = int(match.group(3))
	# normalised_array = test_value/np.amax(test_value)
	# normalised_array = normalised_array.reshape([1,normalised_array.shape[0], 
												# normalised_array.shape[1], 1])
	# pred = np.round(model.predict(normalised_array),0)
	# if pred==1:
		# pixel_int = 0
	# else:
		# pixel_int = 255
	# reconstructed_image[pixel_x][pixel_y]=pixel_int
	# i+=1

# pred_image = np.array(reconstructed_image)
# true_image = test_image

# cv2.imshow('pred', pred_image)
# cv2.imshow('true', true_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


