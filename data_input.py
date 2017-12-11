# Copy Right Kairos03 2017. All Right Reserved.

import pandas as  pd
import numpy as np

#load the data set
train = pd.read_json("data/processed/train.json")
test = pd.read_json("data/processed/test.json")

# generate the training data
# create 3 bands having HH, HV and avg of both
# train
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], ((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
Y_train = train['is_iceberg']

d_train = np.array([[img for img in X_train], [is_iceberg for is_iceberg in Y_train]])
d_train = np.transpose(d_train, [1, 0])

# test
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], ((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
Y_test = train['is_iceberg']

d_test = np.array([[img for img in X_test], [is_iceberg for is_iceberg in Y_test]])
d_test = np.transpose(d_test, [1, 0])


class Dataset:
    def __init__(self, batch_size):
        self.train = d_train
        self.test = d_test
        self.train_size = self.train.shape[0]
        self.test_size = self.test.shape[0]
        self.batch_size = batch_size
        self.total_batch = int(self.train_size/self.batch_size)
        self.batch_cnt = 0

    def next_batch(self):
        if self.batch_cnt == 0:
            np.random.shuffle(self.train)

        start = self.batch_cnt * self.batch_size
        self.batch_cnt += 1
        end = self.batch_cnt * self.batch_size

        if self.batch_cnt >= self.total_batch:
            self.total_batch = 0

        return self.train[start:end][0], self.train[start:end][1]

    def get_test(self):
        return self.test[:][0], self.test[:][1]


def get_dataset(batch_size):
    return Dataset(batch_size=batch_size)


if __name__ == '__main__':
    d = get_dataset(100)