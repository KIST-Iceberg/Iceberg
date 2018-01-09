# 이미지 확인용 python3 code
# -*- encoding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
plt.rcParams['figure.figsize'] = 10, 10
data = pd.read_json('D:/python/Iceberg/input/train.json')

X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
plt.imshow(X_band_1[2,:,:], cmap='gray', interpolation='nearest')