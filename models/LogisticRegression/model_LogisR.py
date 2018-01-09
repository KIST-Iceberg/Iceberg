'''
2018 copyright quisutdeus7. All Right Reserved
# brief : make neural network(Logistic regression) 
'''
# -*- coding : utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

def make_model(X_train, Y_train, A, X_test, Y_test, trainable = True):
"""
make neural network(Logistic regression)
:param X: input data(75*75*9)
:param Y: output data(2)
:param A : Iceberg_angle, band_max, band_variance 
:param trainable : train step is True. if not, it's False.
:return: 
"""    
    with tf.name_scope('Logistic reg'):
        logreg = LogisticRegression()
        logreg.fit(X_train, Y_train)

        predicts = logreg.predict(X_test)
        predict_prob = logreg.predict_proba(X_test)[:,1]
        logloss = log_loss(Y_test, predict_prob)

        train_acc = logreg.score(X_train, Y_train)
        test_acc = logreg.score(X_test, Y_test)

        tf.summary.scalar('log_loss', logloss)
        tf.summary.scalar('train_acc', train_acc)
        tf.summary.scalar('test_acc', test_acc)

    return logloss, mean_acc, predicts