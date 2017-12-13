# Copy Right Kairos03 2017. All Right Reserved.

import pandas as pd
import numpy as np

# load the data set
train = pd.read_json("data/processed/train.json")


# test = pd.read_json("data/processed/test.json")


# generate the training data
# create 3 bands having HH, HV and avg of both
def gen_new_data(data_list):
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data_list["band_1"]])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data_list["band_2"]])
    X = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],
                        ((X_band_1 + X_band_2) / 2)[:, :, :, np.newaxis]], axis=-1)
    Y = np.eye(2)[np.asarray(train['is_iceberg'])]

    return X, Y


# train
x_train, x_label = gen_new_data(train)


# d_test = gen_new_data(test)


class Dataset:
    def __init__(self, batch_size):
        self.train_data = x_train
        self.train_label = x_label
        # self.test = x_label
        self.train_size = self.train_data.shape[0]
        # self.test_size = self.test.shape[0]
        self.batch_size = batch_size
        self.total_batch = int(self.train_size / self.batch_size)
        self.batch_cnt = 0

    def next_batch(self):
        if self.batch_cnt == 0:
            np.random.shuffle(self.train_data)

        start = self.batch_cnt * self.batch_size
        self.batch_cnt += 1
        end = self.batch_cnt * self.batch_size

        if self.batch_cnt >= self.total_batch:
            self.batch_cnt = 0

        x = self.train_data[start:end]
        y = self.train_label[start:end]
        return x, y

    # def get_test(self):
    #     return self.test[:][0], self.test[:][1]


def get_dataset(batch_size):
    return Dataset(batch_size=batch_size)


if __name__ == '__main__':
    d = get_dataset(100)
    for e in range(5):
        for b in range(16):
            xs, ys = d.next_batch()
            print(e, b, xs.shape, ys.shape)
