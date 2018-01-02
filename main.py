# Copy Right Kairos03 2017. All Right Reserved.

import train
import test


if __name__ == '__main__':
    train.train(is_valid=True)
    test.test(train.log_train_path, is_valid=True)
