# Copy Right Kairos03 2017. All Right Reserved.

import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from data import process

# All file path is deprecated
origin_path = 'data/processed/train.json'

pp_x_path = 'data/processed/pp_train_x.pkl'
pp_label_path = 'data/processed/pp_train_label.pkl'
pp_idx_path = "data/processed/pp_train_idx.pkl"

test_path = "data/processed/test.json"
pp_test_path = "data/processed/pp_test_x.pkl"
pp_test_idx_path = "data/processed/pp_test_idx.pkl"


class Dataset:
    def __init__(self, batch_size, data, label, angle, is_shuffle=False, is_valid=False):
        self.data = data
        self.label = label
        self.angle = angle

        self.valid_data = None
        self.valid_label = None
        self.valid_angle = None

        self.data_size = self.data.shape[0]
        print(self.data_size)
        self.batch_size = batch_size
        self.total_batch = int(self.data_size / self.batch_size) + 1
        self.batch_cnt = 0

        self.is_shuffle = is_shuffle
        self.is_valid = is_valid

        if is_valid:
            self.data, self.valid_data, self.label, self.valid_label, self.angle, self.valid_angle = train_test_split(self.data,
                                                                                                                      self.label,
                                                                                                                      self.angle,
                                                                                                                      test_size=0.33,
                                                                                                                      random_state=486)

            self.data_size = self.data.shape[0]
            self.valid_size = self.valid_data.shape[0]

            self.total_batch = int(self.data_size / self.batch_size) + 1
            self.valid_total_batch = int(self.valid_size / self.batch_size) + 1

    def next_batch(self, seed, valid_set=False):

        if valid_set:
            data = self.valid_data
            label = self.valid_label
            angle = self.valid_angle
            total_batch = self.valid_total_batch
        else:
            data = self.data
            label = self.label
            angle = self.angle
            total_batch = self.total_batch

        # shuffle
        if self.is_shuffle and self.batch_cnt == 0:
            np.random.seed(seed)
            np.random.shuffle(data)
            np.random.seed(seed)
            np.random.shuffle(label)
            np.random.seed(seed)
            np.random.shuffle(angle)

        start = self.batch_cnt * self.batch_size
        self.batch_cnt += 1

        if self.batch_cnt == total_batch:
            end = None
        else:
            end = self.batch_cnt * self.batch_size

        xs = data[start:end]
        ys = label[start:end]
        ans = angle[start:end]

        if self.batch_cnt >= total_batch:
            self.batch_cnt = 0

        return xs, ys, ans

    # def get_test(self):
    #     return self.test[:][0], self.test[:][1]


def get_dataset(batch_size, data, label, angle, is_shuffle, is_valid):
    return Dataset(batch_size=batch_size, data=data, label=label, angle=angle, is_shuffle=is_shuffle, is_valid=is_valid)


def load_data():
    """
    load data

    :DEPRECATED:
    Use data/process.load_from_pickle()

    """
    return pickle.load(open(pp_x_path, 'rb')), \
        pickle.load(open(pp_label_path, 'rb')), \
        pickle.load(open(pp_idx_path, 'rb'))


def load_test():
    return pickle.load(open(pp_test_path, 'rb')), pickle.load(open(pp_test_idx_path, 'rb'))


if __name__ == '__main__':

    # deprecated
    # x, y, idx = load_data()

    x, y, angle = process.load_from_pickle()

    print('y', y.shape)
    print('angle', angle.shape)
    dd = get_dataset(1000, x, y, angle, is_shuffle=False, is_valid=True)

    print(dd.valid_data.shape)
    print(dd.data.shape)

    for e in range(2):
        for b in range(dd.total_batch):
            xss, yss, ans = dd.next_batch(valid_set=False)
            print(e, b, xss.shape, yss.shape, ans.shape)

    for e in range(2):
        for b in range(dd.valid_total_batch):
            xss, yss, ans = dd.next_batch(valid_set=True)
            print(e, b, xss.shape, yss.shape, ans.shape)
