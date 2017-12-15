# Copy Right Kairos03 2017. All Right Reserved.

import numpy as np
import pickle


origin_path = 'data/processed/train.json'
pp_x_path = 'data/processed/pp_train_x.pkl'
pp_label_path = 'data/processed/pp_train_label.pkl'
test_path = "data/processed/test.json"


class Dataset:
    def __init__(self, batch_size, x, y):
        self.train_data = x
        self.train_label = y
        # self.test = x_label
        self.train_size = self.train_data.shape[0]
        # self.test_size = self.test.shape[0]
        self.batch_size = batch_size
        self.total_batch = int(self.train_size / self.batch_size)
        self.batch_cnt = 0

    def next_batch(self):
        if self.batch_cnt == 0:
            pass
            # np.random.shuffle(self.train_data)

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


def get_dataset(batch_size, x, y):
    return Dataset(batch_size=batch_size, x=x, y=y)


def load_data():
    return pickle.load(open(pp_x_path, 'rb')), pickle.load(open(pp_label_path, 'rb'))


if __name__ == '__main__':

    x, y = load_data()

    dd = get_dataset(100, x, y)

    for e in range(5):
        for b in range(dd.total_batch):
            xs, ys = dd.next_batch()
            print(e, b, xs.shape, ys.shape)


