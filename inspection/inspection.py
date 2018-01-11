# Copyright Kairos03. All Right Reserved.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

sub1 = pd.read_csv('submissions/bom.csv')
sub2 = pd.read_csv('submissions/test_3_0.7785.csv')
sub3 = pd.read_csv('submissions/sub_fcn.csv')
test = pd.read_json('data/origin/test.json')

total = 0
ids = []
idx = []
for i in range(sub1.shape[0]):
    p1 = sub1.loc[i, 'is_iceberg']
    p2 = sub2.loc[i, 'is_iceberg']
    p3 = sub3.loc[i, 'is_iceberg']
    
    avg = (p1+p2+p3) / 3

    if abs(avg - p2) >= 0.5:
        ids.append(sub1.loc[i, 'id']) 
        idx.append(i)
        total += 1
print('diff', total)

for i in idx:
    img = np.asarray(test.loc[i, 'band_1'])
    img = np.reshape(img, (75, 75))
    plot = plt.imshow(img)
    plt.show()
    