'''
2018 copyright quisutdeus7. All Right Reserved
# brief : make neural network(DNN) 
'''
# -*- coding : utf-8 -*-
import tensorflow as tf
import numpy as np


def var_summary(var):
    """ 
    weight variable summary
    """
    with tf.name_scope('summary'):
        tf.summary.histogram('histogram', var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)

def make_model(X, Y, A, keep_prob, learning_rate):
"""
make neural network(DNN)
"""
 with tf.name_scope('DNN'):
        # concate input data and A
        reshape_X = tf.reshape(X, [-1, 75*75*9])
        X_A = tf.concat([reshape_A, A], 1)
        print('X_A', X_A.shape)

        dense1 = tf.layers.dense(X_A, 5000, tf.nn.relu, name='dense1')
        dense1 = tf.nn.dropout(dense1, keep_prob = keep_prob)

        dense2 = tf.layers.dense(dense1, 500, tf.nn.relu, name='dense2')
        dense2 = tf.nn.dropout(dense2, keep_prob = keep_prob)
        dense3 = tf.layers.dense(dense2, 50, tf.nn.relu, name='dense3')
        dense3 = tf.nn.dropout(dense3, keep_prob = keep_prob)
        dense4 = tf.layers.dense(dense3, 2, tf.nn.relu, name='dense4')

        print('dense1', dense1.shape)
        print('dense2', dense2.shape)
        print('dense3', dense3.shape)
        print('dense4', dense4.shape)

 with tf.name_scope('matrics'):
        with tf.name_scope('xent_func'):
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=dense4))
            tf.summary.scalar('xent', xent)
        with tf.name_scope('opt'):
            opt = tf.train.AdamOptimizer(learning_rate).minimize(xent)
        with tf.name_scope('accuracy'):
            is_correct = tf.equal(tf.argmax(dense4, 1), tf.argmax(Y, 1))
            acc = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='acc')
            tf.summary.scalar('accuracy', acc)
     
        proba = tf.nn.softmax(dense4, name='proba')
        
        return dense4, xent, opt, acc