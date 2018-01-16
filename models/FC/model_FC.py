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
    with tf.name_scope('reshape_concat'):
        # concate input data and A
        bn0 = tf.layers.batch_normalization(X)
        reshape_X = tf.reshape(bn0, (-1, 75*75*9))
        print('[{:s}] \t | {}'.format('reshape', reshape_X.shape))
        X_A = tf.concat((reshape_X, A), axis = 1)
        print('[{:s}] \t | {}'.format('add_data', X_A.shape))
        
    with tf.name_scope('DN1'):
        bn1 = tf.layers.batch_normalization(X_A)
        dense1 = tf.layers.dense(bn1, 1000, tf.nn.relu)
        dense1 = tf.layers.dropout(dense1, rate=keep_prob)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='den1'):
            var_summary(var)
            
        print('[{:s}] \t | {}'.format('dense1', dense1.shape))

    with tf.name_scope('DN2'):
        bn2 = tf.layers.batch_normalization(dense1)
        dense2 = tf.layers.dense(bn2, 200, tf.nn.relu)
        dense2 = tf.layers.dropout(dense2, rate=keep_prob)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='den2'):
            var_summary(var)
            
        print('[{:s}] \t | {}'.format('dense2', dense2.shape))

    with tf.name_scope('DN3'):
        bn3 = tf.layers.batch_normalization(dense2)
        dense3 = tf.layers.dense(bn3, 2, tf.nn.relu)
        dense3 = tf.layers.dropout(dense3, rate=keep_prob)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='den3'):
            var_summary(var)
        
        print('[{:s}] \t | {}'.format('dense3', dense3.shape))

    with tf.name_scope('matrices'):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=dense3), name='xent')
        tf.summary.scalar('xent', xent)
    
        opt = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(xent)
    
        is_correct = tf.equal(tf.argmax(dense3, 1), tf.argmax(Y, 1))
        acc = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', acc)
        
        proba = tf.nn.softmax(dense3, name='proba')
        
    return dense3, xent, opt, acc