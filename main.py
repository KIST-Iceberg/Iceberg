# Copy Right Kairos03 2017. All Right Reserved.
"""
main
"""

import train
import test
from data import process

def main(valid=True):
    
    train.train(valid)
    test.test(train.LOG_TRAIN_PATH, is_test=(not valid))

if __name__ == '__main__':
    # main(True)
    process.main(is_test=False)
    process.main(is_test=True)
    main(False)
