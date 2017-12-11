##!/usr/bin/python
# Original code from https://goo.gl/9MJAZS
# -*- encoding: UTF-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import datetime

from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pylab
plt.rcParams['figure.figsize'] = 10, 10
#%matplotlib inline

#Load the data.
train = pd.read_json("/input/train.json")
test  = pd.read_json("/input/test.json")

#Generate the training data
#Create 3 bands having HH, HV and avg of both
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
# HH, HV, avg value
X_train  = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
Test = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)

is_iceberg_train = train['is_iceberg']
ID = test['ID']

#Import Keras.
from matplotlib import pyplot

# tensorboard
def var_summary(name, var):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)


# Building the model
# HH, HV, avg로 총 3가지
X = tf.placeholder(tf.float32, [None, 75, 75, 3])
Y = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)

# conv Layer 1
W1 = tf.get_variable('W1', [4, 4, 3, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
b1 = tf.get_variable('b1', [64], initializer=tf.truncated_normal_initializer(stddev=0.02))

L1 = tf.nn.conv2d(input=X, filter=W1, strides=[1, 1, 1, 1], padding='VALID')
L1 = L1 + b1
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
L1 = tf.nn.dropout(L1, keep_prob)

# Conv Layer 2
W2 = tf.get_variable('W2', [3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
b2 = tf.get_variable('b2', [128], initializer=tf.truncated_normal_initializer(stddev=0.02))
L2 = tf.nn.conv2d(input=L1, filter=W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = L2 + b2
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob)

#Conv Layer 3
W3 = tf.get_variable('W3', [3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
b3 = tf.get_variable('b3', [256], initializer=tf.truncated_normal_initializer(stddev=0.02))
L3 = tf.nn.conv2d(input=L2, filter=W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = L3 + b3
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob)

#Conv Layer 4
W4 = tf.get_variable('W4', [3, 3, 256, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
b4 = tf.get_variable('b4', [512], initializer=tf.truncated_normal_initializer(stddev=0.02))
L4 = tf.nn.conv2d(input=L3, filter=W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = L4 + b4
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
L4 = tf.nn.dropout(L4, keep_prob)

# FC Layer1
W5 = tf.Variable(tf.random_normal([3 * 3 * 512, 512], stddev=0.02))
L5 = tf.reshape(L4, [-1, 3 * 3 * 512])
L5 = tf.matmul(L5, W5) # TODO : 이부분이 필요한가? 그냥 reshape만 해서 나와도 되지 않나?
L5 = tf.nn.relu(L5)
L5 = tf.nn.dropout(L5, keep_prob)

# FC Layer2
W6 = tf.Variable(tf.random_normal([512, 256], stddev=0.02))
L6 = tf.matmul(L5, W6)
L6 = tf.nn.relu(L6)
L6 = tf.nn.dropout(L6, keep_prob)

# FC Layer3
W7 = tf.Variable(tf.random_normal([256, 1], stddev=0.02))
L7 = tf.matmul(L6, W7)
model = tf.nn.sigmoid(L7)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08). minimize(loss)

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#########
# 신경망 모델 학습

with tf.Session() as sess:

    # random_state : 이전 raondom하게 split한 값들의 그룹명? 느낌으로 값 자체에는 의미가 없음
    X_train_cv, X_test, is_ceberg_cv, is_iceberg_test = train_test_split(X_train, is_iceberg_train, random_state=1, train_size=0.75)

    batch_size = 24
    total_batch = int(len(X_train_cv)/batch_size)
    n_conv2d = 4
    n_fc = 2
    log_dir = ("tensorboard/n_L:%d c_FC:%d time:" % (n_conv2d, n_fc) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    for epoch in range():
        total_loss = 0

        for a in range(total_batch):
             batch_xs, batch_ys = next_batch(batch_size, X_train_cv, is_ceberg_cv)
             _, acc, loss_val = sess.run([optimizer, loss],
                                        feed_dict={X:X_train_cv,
                                                   Y:is_ceberg_cv,
                                                   keep_prob:0.2})
             total_loss += loss_val
        print('Epoch:', '%04d' % (epoch + 1),
              'Avg. loss =', '{:.3f}'.format(total_loss / total_batch))

    saver.save(sess, 'tmp/Iceberg_model.ckpt')
