# Copy Right Kairos03 2017. All Right Reserved.
"""
main
"""

import train
import test

def main(valid=True):
    train.train(valid)
    test.test(train.LOG_TRAIN_PATH, is_test=(not valid))

if __name__ == '__main__':
    main(True)
