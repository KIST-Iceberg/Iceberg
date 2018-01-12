# Copyright Kairos03. All Right Reserved.

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale

from data import process

files = os.listdir('submissions')
subs = []
for file in files:
    path = 'submissions/'+file
    subs.append(pd.read_csv(path))
    print('[READ] ', 'submissions/'+file)

final = pd.DataFrame()
final['id'] = subs[0]['id']
final['is_iceberg'] = np.exp(np.mean(
    [sub['is_iceberg'].apply(lambda x: np.log(x)) for sub in subs], axis=0))
final.to_csv('submissions/ensamble.csv', index=False, float_format='%.6f')

del files

total = 0
idx = []
pro = []
# test = pd.read_hdf('data/origin/test.h5', 'df')
train = pd.read_json('data/origin/train.json')

for i in range(final['id'].shape[0]):
    probs = []
    for sub in subs:
        probs.append(sub.loc[i, 'is_iceberg'])
    
    mean = np.mean(probs)
    for p in probs:
        if abs(mean - p) > 0.5:
            idx.append(i)
            pro.append(probs)
            total += 1
            break
    
print('diff', total)

# for i in idx:
for i in range(50):
    print(final.loc[i,'id'], np.round(pro[i], 2), np.round(final.loc[i, 'is_iceberg'], 2))
    
    # original
    img1 = np.asarray(train.loc[i, 'band_1'])
    img1 = np.reshape(img1, (75, 75))

    img2 = np.asarray(train.loc[i, 'band_2'])
    img2 = np.reshape(img2, (75, 75))

    img3 = (img1 + img2)/2
    img3 = np.reshape(img3, (75, 75))

    img4 = -abs(img1)-abs(img2)
    img4 = np.reshape(img4, (75, 75))

    def iso(arr, rate=2):
        p = np.reshape(np.array(arr), [75,75]) > (np.mean(arr)+rate*np.std(arr))
        return p * np.reshape(np.array(arr), [75,75])

    img5 = iso(img1, 2)
    img6 = iso(img1, 1.9)
    img7 = iso(img1, 1.8)
    img8 = iso(img1, 1.7)

    fig=plt.figure(figsize=(12, 6))
    fig.suptitle("id_" + final.loc[i,'id'] + '  is_iceberg_' + str(train.loc[i, 'is_iceberg']))

    fig.add_subplot(3, 4, 1).set_title('band_1')
    plt.imshow(img1)
    fig.add_subplot(3, 4, 2).set_title('band_2')
    plt.imshow(img2)
    fig.add_subplot(3, 4, 3).set_title('mean(band_1 + band2)')
    plt.imshow(img3)
    fig.add_subplot(3, 4, 4).set_title('- |band_1| - |band2|')
    plt.imshow(img4)

    fig.add_subplot(3, 4, 5).set_title('- |band_1| - |band2|')
    plt.imshow(img5)
    fig.add_subplot(3, 4, 6).set_title('- |band_1| - |band2|')
    plt.imshow(img6)
    fig.add_subplot(3, 4, 7).set_title('- |band_1| - |band2|')
    plt.imshow(img7)
    


    # crop
    # center pos
    temp = img3
    for i in range(75):
        for j in range(75):
            if (i<7 or i>68) or (j<7 or j>68): 
                temp[i, j] = -30

    fig.add_subplot(3, 4, 8).set_title('- |band_1| - |band2|')
    plt.imshow(temp)

    cx = temp.argmax() // 75
    cy = temp.argmax() % 75
    img_crop = process.crop_center(img5, cx, cy, size=40)

    fig.add_subplot(3, 4, 9).set_title('crop_original')
    plt.imshow(img_crop)

    fig.add_subplot(3, 4, 10).set_title('crop_gray')
    plt.imshow(img_crop, cmap='gray')

    img_crop = scale(img_crop, with_std=False)
    fig.add_subplot(3, 4, 11).set_title('crop_gamma')
    plt.imshow(np.power(img_crop, 2))

    fig.add_subplot(3, 4, 12).set_title('crop_gamma_gray')
    plt.imshow(np.power(img_crop, 2), cmap='gray')

    plt.show()
    