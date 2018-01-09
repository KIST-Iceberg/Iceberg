# Code snippets for image preprocessing test

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from skimage.filters import gaussian
#from skimage.segmentation import random_walker
from skimage.segmentation import active_contour

train = pd.read_json('input/train.json')
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')

def get_stats(train,label=1):
    train['max'+str(label)] = [np.max(np.array(x)) for x in train['band_'+str(label)] ]
    train['maxpos'+str(label)] = [np.argmax(np.array(x)) for x in train['band_'+str(label)] ]
    train['min'+str(label)] = [np.min(np.array(x)) for x in train['band_'+str(label)] ]
    train['minpos'+str(label)] = [np.argmin(np.array(x)) for x in train['band_'+str(label)] ]
    train['med'+str(label)] = [np.median(np.array(x)) for x in train['band_'+str(label)] ]
    train['std'+str(label)] = [np.std(np.array(x)) for x in train['band_'+str(label)] ]
    train['mean'+str(label)] = [np.mean(np.array(x)) for x in train['band_'+str(label)] ]
    train['p25_'+str(label)] = [np.sort(np.array(x))[int(0.25*75*75)] for x in train['band_'+str(label)] ]
    train['p75_'+str(label)] = [np.sort(np.array(x))[int(0.75*75*75)] for x in train['band_'+str(label)] ]
    train['mid50_'+str(label)] = train['p75_'+str(label)]-train['p25_'+str(label)]

    return train
    
def decibel_to_linear(band):
     # convert to linear units
    return np.power(10,np.array(band)/10)    

def plot_var(name,nbins=50):
    minval = train[name].min()
    maxval = train[name].max()
    plt.hist(train.loc[train.is_iceberg==1,name],range=[minval,maxval],
             bins=nbins,color='b',alpha=0.5,label='Boat')
    plt.hist(train.loc[train.is_iceberg==0,name],range=[minval,maxval],
             bins=nbins,color='r',alpha=0.5,label='Iceberg')
    plt.legend()
    plt.xlim([minval,maxval])
    plt.xlabel(name)
    plt.ylabel('Number')
    plt.show()    
    
NofSample = 12

icebergs = train[train.is_iceberg==1].sample(n=NofSample,random_state=123)
ships = train[train.is_iceberg==0].sample(n=NofSample,random_state=457)

def getHigh(img, length):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    #magnitude_spectrum = 20*np.log(np.abs(fshift))

    rows = np.size(img, 0) #taking the size of the image
    cols = np.size(img, 1)
    crow, ccol = rows/2, cols/2
    
    fshift[crow-length:crow+length, ccol-length:ccol+length] = 0
    f_ishift= np.fft.ifftshift(fshift)

    img_back = np.fft.ifft2(f_ishift) ## shift for centering 0.0 (x,y)
    
    return np.abs(img_back)
    
def getLow(img, length):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    #magnitude_spectrum = 20*np.log(np.abs(fshift))

    rows = np.size(img, 0) #taking the size of the image
    cols = np.size(img, 1)
    crow, ccol = rows/2, cols/2
    
    fshift[0:length, :] = 0
    fshift[:, 0:length] = 0
    
    fshift[rows-length:rows, :] = 0
    fshift[:, cols-length:cols] = 0
    
    f_ishift= np.fft.ifftshift(fshift)

    img_back = np.fft.ifft2(f_ishift) ## shift for centering 0.0 (x,y)
    
    return np.abs(img_back)    
    
def getHigh2(img, length=1):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows = np.size(img, 0) #taking the size of the image
    cols = np.size(img, 1)
    crow, ccol = rows/2, cols/2
    
    fshift[crow-length:crow+length, ccol-length:ccol+length] = 0
    f_ishift= np.fft.ifftshift(fshift)

    img_back = np.power(np.abs(np.fft.ifft2(f_ishift)), 2) # Gamma correction
    img_back = np.abs(np.fft.ifft2(f_ishift)) ## shift for centering 0.0 (x,y)
    
    if np.std(img_back) > 0.05:
        img_back = np.power(img_back, 2)
    
    img_back = (img_back - np.mean(img_back)) / np.std(img_back)
    img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
    
    return img_back
    
def cropROI(img):
    highImg = getHigh(img, 1)
    intThreshold = np.mean(np.sum(highImg, 0)) * 1.1

    sumRows = np.sum(highImg, 0) > intThreshold
    sumCols = np.sum(highImg, 1) > intThreshold

    sumRows = np.asmatrix(sumRows)
    sumCols = np.transpose(np.asmatrix(sumCols))

    ROIMask = np.matmul(sumCols, sumRows)

    return np.multiply(highImg, ROIMask)

def dispResult():
    for i in range(NofSample):
        #print(icebergs.iloc[i, 3]) #print angle
        
        iceimg1 = np.reshape(np.array(icebergs.iloc[i,0]),(75,75))
        
        plt.subplot(331)
        plt.imshow(iceimg1, cmap='gray')

        plt.subplot(332)
        plt.imshow(decibel_to_linear(iceimg1), cmap='gray')
        
        plt.subplot(333)
        plt.imshow(getHigh2(iceimg1), cmap='gray')
        #print(np.std(getHigh2(decibel_to_linear(iceimg1))))
        iceimg2 = np.reshape(np.array(icebergs.iloc[i,1]),(75,75))
        
        plt.subplot(334)
        plt.imshow(iceimg2, cmap='gray')

        plt.subplot(335)
        plt.imshow(decibel_to_linear(iceimg2), cmap='gray')
        
        plt.subplot(336)
        plt.imshow(getHigh2(iceimg2), cmap='gray')
        
        multimg = np.multiply(abs(iceimg1), abs(iceimg2))
        plt.subplot(337)
        plt.imshow(multimg, cmap='gray')
        
        plt.subplot(338)
        #plt.hist(np.reshape(multimg, (75*75, 1)), range=[np.min(multimg), np.max(multimg)], bins=10)
        plt.imshow(np.multiply(decibel_to_linear(iceimg1),decibel_to_linear(iceimg2)), cmap='gray')
        
        #multimg = getHigh2(multimg)
        plt.subplot(339)
        plt.imshow(getHigh2(multimg), cmap='gray')
        #plt.hist(np.reshape(multimg, (75*75, 1)), range=[np.min(multimg), np.max(multimg)], bins=10)

        plt.show()
        plt.waitforbuttonpress(0)


def writeResult():
    i = 1
    
    iceimg1 = np.reshape(np.array(icebergs.iloc[i,0]),(75,75))
    iceimg2 = np.reshape(np.array(icebergs.iloc[i,1]),(75,75))
    
    # Original
    ax = plt.subplot(131)
    ax.imshow(iceimg1, cmap='gray')
    ax.set_title('Band 1')

    ax = plt.subplot(132)
    ax.imshow(iceimg2, cmap='gray')
    ax.set_title('Band 2')
    
    multimg = np.multiply(abs(iceimg1), abs(iceimg2))
    ax = plt.subplot(133)
    ax.imshow(multimg, cmap='gray')
    ax.set_title('Combined')
    
    plt.savefig('img_bands.png', format='png')
    
    # Decibel to Linear
    ax = plt.subplot(131)
    ax.imshow(decibel_to_linear(iceimg1), cmap='gray')
    ax.set_title('Band 1')

    ax = plt.subplot(132)
    ax.imshow(decibel_to_linear(iceimg2), cmap='gray')
    ax.set_title('Band 2')
    
    ax = plt.subplot(133)
    ax.imshow(np.multiply(decibel_to_linear(iceimg1),decibel_to_linear(iceimg2)), cmap='gray')
    ax.set_title('Combined')
    
    plt.savefig('img_linear.png', format='png')
    
    # High-pass Filtering
    ax = plt.subplot(131)
    ax.imshow(getHigh2(iceimg1), cmap='gray')
    ax.set_title('Band 1')

    ax = plt.subplot(132)
    ax.imshow(getHigh2(iceimg2), cmap='gray')
    ax.set_title('Band 2')
    
    ax = plt.subplot(133)
    ax.imshow(getHigh2(multimg), cmap='gray')
    ax.set_title('Combined')

    plt.savefig('img_high.png', format='png')
    
def rotResult():
    for i in range(NofSample):
        iceimg1 = getHigh2(np.reshape(np.array(icebergs.iloc[i,0]),(75,75)))
        iceimg2 = getHigh2(np.reshape(np.array(icebergs.iloc[i,1]),(75,75)))
        iceimg3 = getHigh2(np.multiply(abs(np.reshape(np.array(icebergs.iloc[i,0]),(75,75))), np.reshape(np.array(icebergs.iloc[i,1]),(75,75))))

        # Testing with combined channel
        img = iceimg3
        rows,cols = img.shape
        
        # Original image
        plt.subplot(3,4,1)
        plt.imshow(img)
        
        # For 360 degree
        for j in range(1,12):
            # Rotation matrix
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 30*j, 1)
            
            # Apply matrix
            dst = cv2.warpAffine(img,M,(cols,rows))
            
            # Fill the blank with original pixels
            dst = np.where(dst == 0, img, dst)

            plt.subplot(3, 4, j+1)
            plt.imshow(dst)
            
        plt.show()
        plt.waitforbuttonpress(0)
            
#dispResult()
#writeResult()
rotResult()


