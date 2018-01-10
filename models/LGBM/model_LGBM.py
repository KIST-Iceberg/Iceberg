'''
2018 copyright quisutdeus7. All Right Reserved
# brief : make neural network(LGBM) 
'''
# -*- coding : utf-8 -*-
import tensorflow as tf
import numpy as np

def make_model(X, Y, A, trainable = True):
"""
make neural network(LGBM)
:param X: input data(75*75*9)
:param Y: output data(2)
:param A : Iceberg_angle, band_max, band_variance 
:param trainable : train step is True. if not, it's False.
:return: dense4, xent, accuracy
"""    