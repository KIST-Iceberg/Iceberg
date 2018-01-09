#!/usr/bin/python

# From https://goo.gl/9MJAZS
import cv2
import numpy as np
import pandas as pd

from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pylab

# High-pass filtering & Gamma correction
def getHigh(img, length=1):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)

	rows = np.size(img, 0) #taking the size of the image
	cols = np.size(img, 1)
	crow, ccol = rows/2, cols/2
	
	fshift[crow-length:crow+length, ccol-length:ccol+length] = 0
	f_ishift= np.fft.ifftshift(fshift)

	img_back = np.power(np.abs(np.fft.ifft2(f_ishift)), 2) ## shift for centering 0.0 (x,y)
	
	img_back = (img_back - np.mean(img_back)) / np.std(img_back)
	img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
	
	return img_back

#Import Keras.
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

#define our model
def getModel():
	#Building the model
	gmodel=Sequential()
	#Conv Layer 1
	gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
	gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	gmodel.add(Dropout(0.2))

	#Conv Layer 2
	gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))

	#Conv Layer 3
	gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))

	#Conv Layer 4
	gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))

	#Flatten the data for upcoming dense layers
	gmodel.add(Flatten())

	#Dense Layers
	gmodel.add(Dense(512))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(0.2))

	#Dense Layer 2
	gmodel.add(Dense(256))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(0.2))

	#Sigmoid Layer
	gmodel.add(Dense(1))
	gmodel.add(Activation('sigmoid'))

	mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	gmodel.compile(loss='binary_crossentropy',
								optimizer=mypotim,
								metrics=['accuracy'])
	gmodel.summary()
	return gmodel

file_path = ".model_weights_imgproc.hdf5"

gmodel = getModel()
gmodel.load_weights(filepath=file_path)

test  = pd.read_json("input/test.json")

X_band_test_1 = np.array([getHigh(np.array(band).astype(np.float32).reshape(75, 75)) for band in test["band_1"]])
X_band_test_2 = np.array([getHigh(np.array(band).astype(np.float32).reshape(75, 75)) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis], X_band_test_2[:, :, :, np.newaxis], ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)

predicted_test=gmodel.predict_proba(X_test)

submission = pd.DataFrame()
submission['id'] = test['id']
submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('sub.csv', index=False)

