#!/usr/bin/python

# Original code: https://goo.gl/9MJAZS
# Modified for preprocessing

import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from CyclicLR import CyclicLR

import pylab
plt.rcParams['figure.figsize'] = 10, 10
#%matplotlib inline

#Load the data.
train = pd.read_json("input/train.json")
test  = pd.read_json("input/test.json")

# High-pass filtering & Gamma correction
def getHigh(img, length=1):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)

	rows = np.size(img, 0) #taking the size of the image
	cols = np.size(img, 1)
	crow, ccol = int(rows/2), int(cols/2)
	
	fshift[crow-length:crow+length, ccol-length:ccol+length] = 0
	f_ishift= np.fft.ifftshift(fshift)

	img_back = np.power(np.abs(np.fft.ifft2(f_ishift)), 2) ## shift for centering 0.0 (x,y)
	
	img_back = (img_back - np.mean(img_back)) / np.std(img_back)
	img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
	
	return img_back

#Generate the training data
train.inc_angle = train.inc_angle.replace('na',0)
idx_meaningful = np.where(train.inc_angle>0)

#Create 3 bands having HH, HV and avg of both
X_band_1 = np.array([getHigh(np.array(band).astype(np.float32).reshape(75, 75)) for band in train["band_1"]])
X_band_2 = np.array([getHigh(np.array(band).astype(np.float32).reshape(75, 75)) for band in train["band_2"]])
X_train  = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)

X_train = X_train[idx_meaningful[0],...]

# Flips
def get_more_images(imgs):
	
	more_images = []
	vert_flip_imgs = []
	hori_flip_imgs = []
	  
	for i in range(0,imgs.shape[0]):
		a=imgs[i,:,:,0]
		b=imgs[i,:,:,1]
		c=imgs[i,:,:,2]
		
		av=cv2.flip(a,1)
		ah=cv2.flip(a,0)
		bv=cv2.flip(b,1)
		bh=cv2.flip(b,0)
		cv=cv2.flip(c,1)
		ch=cv2.flip(c,0)
		
		vert_flip_imgs.append(np.dstack((av, bv, cv)))
		hori_flip_imgs.append(np.dstack((ah, bh, ch)))
	  
	v = np.array(vert_flip_imgs)
	h = np.array(hori_flip_imgs)
	   
	more_images = np.concatenate((imgs,v,h))
	
	return more_images
	
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
	dropoutRate = 0.2
	#Building the model
	gmodel=Sequential()
	#Conv Layer 1
	gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
	gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	gmodel.add(Dropout(dropoutRate))

	#Conv Layer 2
	gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(dropoutRate))

	#Conv Layer 3
	gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(dropoutRate))

	#Conv Layer 4
	gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(dropoutRate))

	#Flatten the data for upcoming dense layers
	gmodel.add(Flatten())

	#Dense Layers
	gmodel.add(Dense(512))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(dropoutRate))

	#Dense Layer 2
	gmodel.add(Dense(256))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(dropoutRate))

	#Sigmoid Layer
	gmodel.add(Dense(1))
	gmodel.add(Activation('sigmoid'))

	mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	gmodel.compile(loss='binary_crossentropy',
								optimizer=mypotim,
								metrics=['accuracy'])
	gmodel.summary()
	return gmodel


def get_callbacks(filepath, patience=2):
	es = EarlyStopping('val_loss', patience=patience, mode="min")
	msave = ModelCheckpoint(filepath, save_best_only=True)
	# step_size = 2-8 x # of training iterations in an epoch = 2 * 113 = 226
	clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=230., mode='exp_range', gamma=0.99994) # initial step_size = 2000
	return [es, msave, clr]
		
file_path = ".model_weights_imgproc.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

Y_train = train['is_iceberg']	
Y_train = Y_train[idx_meaningful[0]]

X_train_cv, X_valid, Y_train_cv, Y_valid = train_test_split(X_train, Y_train, random_state=1, train_size=0.75)

Xtr_more = get_more_images(X_train_cv) 
Ytr_more = np.concatenate((Y_train_cv,Y_train_cv,Y_train_cv))

import os

gmodel = getModel()
gmodel.fit(Xtr_more, Ytr_more, batch_size=32, epochs=50, verbose=1, validation_data=(X_valid, Y_valid), callbacks=callbacks)
#gmodel.fit(X_train_cv, Y_train_cv, batch_size=24, epochs=50, verbose=1, validation_data=(X_valid, Y_valid), callbacks=callbacks)
#gmodel.fit(Xtr_more, Ytr_more, batch_size=32, epochs=50, verbose=1, callbacks=callbacks, validation_split=0.25)
					
gmodel.load_weights(filepath=file_path)
score = gmodel.evaluate(X_train_cv, Y_train_cv, verbose=1)

print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = gmodel.evaluate(X_valid, Y_valid, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])



