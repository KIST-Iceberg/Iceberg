# Copy Right Kairos03 2017. All Right Reserved.
"""
train
"""
from __future__ import absolute_import

import time

import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss

from models import model_conv_simple
from data import process
from data import data_input

# Hyper-parameters
LEARNING_RATE = 1e-3
TOTAL_EPOCH = 500
BATCH_SIZE = 1000
DROPOUT_RATE = 0.2
REGULARIZATION_BETA = 1e-2
RANDOM_SEED = int(np.random.random() * 100)

CURRENT = time.time()

SESSION_NAME = '{}_lr{}_ep{}_b{}'.format(
    time.ctime(), LEARNING_RATE, TOTAL_EPOCH, BATCH_SIZE)
LOG_TRAIN_PATH = './log/' + SESSION_NAME + '/train/'
LOG_TEST_PATH = './log/' + SESSION_NAME + '/test/'
MODEL_PATH = LOG_TRAIN_PATH + 'model.ckpt'


def train(is_valid):
    # data set load
    x, y, angle = process.load_from_pickle()

    inputs = data_input.get_dataset(
        BATCH_SIZE, x, y, angle, is_shuffle=True, is_valid=is_valid)

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 75, 75, 3], name='X')
        Y = tf.placeholder(tf.float32, [None, 2], name='Y')
        A = tf.placeholder(tf.float32, [None, 1], name='A')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    xent, optimizer, accuracy = model_conv_simple.make_model(
        X, Y, A, keep_prob, LEARNING_RATE)

    with tf.name_scope('hyperparam'):
        tf.summary.scalar('learning_rate', LEARNING_RATE)
        tf.summary.scalar('batch_size', BATCH_SIZE)
        tf.summary.scalar('dropout_rate', DROPOUT_RATE)
        tf.summary.scalar('regularization_beta', REGULARIZATION_BETA)
        tf.summary.scalar('random_seed', RANDOM_SEED)

    print('Train Start')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        # last_time = CURRENT

        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, sess.graph)
        test_writer = tf.summary.FileWriter(LOG_TEST_PATH)
        tf.global_variables_initializer().run()

        total_batch = inputs.total_batch
        if is_valid:
            valid_total_batch = inputs.valid_total_batch

        probability = sess.graph.get_tensor_by_name('matrices/proba:0')

        for epoch in range(TOTAL_EPOCH):

            epoch_loss = epoch_acc = 0
            xs = ys = None

            for batch_num in range(total_batch):
                xs, ys, ans = inputs.next_batch(valid_set=False)

                _, loss, acc = sess.run([optimizer, xent, accuracy],
                                        feed_dict={X: xs, Y: ys, A: ans, keep_prob: DROPOUT_RATE})
                epoch_loss += loss
                epoch_acc += acc

                # display time spend for 20 batch
                # if batch_num % 20 == 0:
                #     print("[{:05.3f}] Batch {} finish.".format(time.time() - last_time, batch_num))
                #     last_time = time.time()

            summary = sess.run(merged,
                               feed_dict={X: xs, Y: ys, A: ans, keep_prob: DROPOUT_RATE})
            train_writer.add_summary(summary, epoch)

            epoch_loss = epoch_loss / total_batch
            epoch_acc = epoch_acc / total_batch
            if epoch % 20 == 9 or epoch == 0:
                print('[{:05.3f}] \tTRAIN EP: {:05d} | \tloss: {:0.5f}| \tacc: {:0.5f}'
                      .format(time.time() - CURRENT, epoch, epoch_loss, epoch_acc))

            # valid
            if is_valid and (epoch % 20 == 9 or epoch == 0):

                epoch_acc = logloss = 0
                xs = ys = None

                for batch_num in range(valid_total_batch):
                    xs, ys, ans = inputs.next_batch(valid_set=True)

                    acc, predict = sess.run([accuracy, probability],
                                            feed_dict={X: xs, Y: ys, A: ans, keep_prob: 1})

                    epoch_acc += acc
                    logloss += log_loss(ys, predict)

                summary = sess.run(merged,
                                   feed_dict={X: xs, Y: ys, A: ans, keep_prob: 1})
                test_writer.add_summary(summary, epoch)

                epoch_acc = epoch_acc / valid_total_batch
                logloss = logloss / valid_total_batch

                print('[{:05.3f}] \tVALID EP: {:05d} | \tloss: {:0.5f}| \tacc: {:0.5f}'
                        .format(time.time() - CURRENT, epoch, logloss, epoch_acc))

        # model save
        saver.save(sess, MODEL_PATH)

    print('Train Finish')


if __name__ == '__main__':
    train(is_valid=True)
    # train(is_valid=False)
