# Copy Right Kairos03 2017. All Right Reserved.

import train
import test


if __name__ == '__main__':
    train.train(is_valid=False)
    test.test(train.LOG_TRAIN_PATH, is_test=True)
