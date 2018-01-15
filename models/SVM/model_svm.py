# Copy Right quisutdeus7 2018. All Right Reserved.
# SVM model
"""
"""
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from data import process

def data_process():
    # load and split data
    x, y, angle = process.load_from_pickle()
    x_train, x_test, y_train, y_test, angle_train, angle_test = train_test_split(x, y, angle)

    print("Total dataset info")
    print("shape| X_train: {}, X_test : {}" .format(x_train.shape, x_test.shape))
    print("shape| Y_train: {}, Y_test : {}" .format(y_train.shape, y_test.shape))
    print("shape| angle  {}" .format(angle.shape))

    return x_train, x_test, y_train, y_test, angle_train, angle_test
def make_model(X_train, X_test, Y_train, Y_test, A_train, A_test):
    # iter number
    max_iter = 10

    # change shape
    layer_train = np.reshape(np.ravel(X_train), (-1, 75*75*9))
    layer_test = np.reshape(np.ravel(X_test), (-1, 75*75*9))
    print('[{:s}] \t | {}' .format('reshape_train', layer_train.shape))
    print('[{:s}] \t\t | {}' .format('reshape_test', layer_test.shape))

    # concate img + angle
    layer_train = np.concatenate((layer_train, A_train), axis=1)
    layer_test = np.concatenate((layer_test, A_test), axis=1)
    print('[{:s}] \t | {}'.format('add_data_train', layer_train.shape))
    print('[{:s}] \t | {}'.format('add_data_test', layer_test.shape))
    Y_train = Y_train[:,0]
    Y_test = Y_test[:,0]

    clf = svm.SVC(max_iter=max_iter, probability=True)
    clf.fit(layer_train, Y_train)
    print('SVC fit complete')

    predicts = clf.predict(layer_test)
    predicts_prob = clf.predict_proba(layer_test)[:,1]
    logloss = log_loss(Y_test, predicts_prob)

    train_acc = clf.score(layer_train, Y_train)
    test_acc = clf.score(layer_test, Y_test)

    print('SVC Result')
    print('[max_itr : {:f} | logloss : {:0.5f} | train_acc : {:0.5f} | test_acc : {:0.5f}]'
           .format(max_iter, logloss, train_acc, test_acc))

if __name__ == '__main__':
    x_tr, x_te, y_tr, y_te, angle_tr, angle_te = data_process()
    make_model(x_tr, x_te, y_tr, y_te, angle_tr, angle_te)
