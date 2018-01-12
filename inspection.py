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
# final.to_csv('submissions/ensamble.csv', index=False, float_format='%.6f')

del files

total = 0
idx = []
pro = []
test = pd.read_hdf('data/origin/test.h5', 'df')

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
for i in idx:
    print(final.loc[i,'id'], np.round(pro[i], 2), np.round(final.loc[i, 'is_iceberg'], 2))
    
    # original
    img1 = np.asarray(test.loc[i, 'band_1'])
    img1 = np.reshape(img1, (75, 75))

    img2 = np.asarray(test.loc[i, 'band_2'])
    img2 = np.reshape(img2, (75, 75))

    img3 =  (abs(img1) * abs(img2))
    img3 = np.reshape(img3, (75, 75))

    img4 = - abs(img1) - abs(img2)
    img4 = np.reshape(img4, (75, 75))

    fig=plt.figure(figsize=(12, 6))
    fig.suptitle("id_" + final.loc[i,'id'])

    fig.add_subplot(2, 4, 1).set_title('band_1')
    plt.imshow(img1)
    fig.add_subplot(2, 4, 2).set_title('band_2')
    plt.imshow(img2)
    fig.add_subplot(2, 4, 3).set_title('|band_1| * |band2|')
    plt.imshow(img3)
    fig.add_subplot(2, 4, 4).set_title('- |band_1| - |band2|')
    plt.imshow(img4)

    # crop
    # center pos
    cx = img.argmax() // 75
    cy = img.argmax() % 75
    img_crop = process.crop_center(img4, cx, cy)

    fig.add_subplot(2, 4, 5).set_title('crop_original')
    plt.imshow(img_crop)

    fig.add_subplot(2, 4, 6).set_title('crop_gray')
    plt.imshow(img_crop, cmap='gray')

    img_crop = scale(img_crop, with_std=False)
    fig.add_subplot(2, 4, 7).set_title('crop_gamma')
    plt.imshow(np.power(img_crop, 2))

    fig.add_subplot(2, 4, 8).set_title('crop_gamma_gray')
    plt.imshow(np.power(img_crop, 2), cmap='gray')

    plt.show()
    