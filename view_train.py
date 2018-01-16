# 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_json('data/origin/train.json')

train_size = train.shape[0]

for i in range((train_size//8) + 1):
    print(i*8, (i+1)*8)
    band1_imgs = train.loc[i*8:(i+1)*8, 'band_1'].reset_index(drop=True)
    band2_imgs = train.loc[i*8:(i+1)*8, 'band_2'].reset_index(drop=True)
    ids = train.loc[i*8:(i+1)*8, 'id'].reset_index(drop=True)

    fig = plt.figure(figsize=(8, 8))
    for j in range(ids.size-1):
        curid = ids[j]
        ax = fig.add_subplot(4, 6, 2*j+1)
        ax.set_title(curid)
        plt.imshow(np.reshape(band1_imgs[j], (75, 75)))
        ax = fig.add_subplot(4, 6, (2*j)+2)
        ax.set_title(curid)
        plt.imshow(np.reshape(band2_imgs[j], (75, 75)))
        
    plt.show()
