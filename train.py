# Copy Right Kairos03 2017. All Right Reserved.

from __future__ import absolute_import

import time

import tensorflow as tf
import numpy as np

from models import conv2_mp2_dense1
from data import process
from data import data_input

# Hyper-parameters
LEARNING_RATE = 1e-3
TOTAL_EPOCH = 150
BATCH_SIZE = 500
DROPOUT_RATE = 0.9

CURRENT = time.time()

SESSION_NAME = '{}_lr{}_ep{}_b{}'.format(
    time.ctime(), LEARNING_RATE, TOTAL_EPOCH, BATCH_SIZE)
LOG_TRAIN_PATH = './log/' + SESSION_NAME + '/train/'
LOG_TEST_PATH = './log/' + SESSION_NAME + '/test/'
MODEL_PATH = LOG_TRAIN_PATH + 'model.ckpt'

def train(is_valid):
    # data set load
    x, y, angle = process.load_from_pickle()

    y = data_input.one_hot(y)
    angle = np.reshape(angle, [-1, 1])

    inputs = data_input.get_dataset(
        BATCH_SIZE, x, y, angle, is_shuffle=True, is_valid=is_valid)

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 75, 75, 3], name='X')
        Y = tf.placeholder(tf.float32, [None, 2], name='Y')
        A = tf.placeholder(tf.float32, [None, 1], name='A')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    model, xent, optimizer, accuracy = conv2_mp2_dense1.make_model(
        X, Y, A, keep_prob, LEARNING_RATE, 0.01)

    print('Train Start')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        last_time = CURRENT

        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, sess.graph)
        test_writer = tf.summary.FileWriter(LOG_TEST_PATH)
        tf.global_variables_initializer().run()

        total_batch = inputs.total_batch
        valid_total_batch = inputs.valid_total_batch

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
            if epoch % 10 == 9 or epoch == 0: 
                print('[{:05.3f}] EP: {:05d}, loss: {:0.5f}, acc: {:0.5f}'
                    .format(time.time() - CURRENT, epoch, epoch_loss, epoch_acc))

            # valid
            if is_valid:

                epoch_acc = 0
                xs = ys = None

                for batch_num in range(valid_total_batch):
                    xs, ys, ans = inputs.next_batch(valid_set=True)

                    acc = sess.run(accuracy,
                                   feed_dict={X: xs, Y: ys, A: ans, keep_prob: 1})

                    epoch_acc += acc

                summary = sess.run(merged,
                                   feed_dict={X: xs, Y: ys, A: ans, keep_prob: 1})
                test_writer.add_summary(summary, epoch)

                epoch_acc = epoch_acc / valid_total_batch
                if epoch % 10 == 9 or epoch == 0: 
                    print('[{:05.3f}] VALID EP: {:05d}, acc: {:0.5f}'
                        .format(time.time() - CURRENT, epoch, epoch_acc))

        # model save
        saver.save(sess, MODEL_PATH)

    print('Train Finish')


if __name__ == '__main__':
    train(is_valid=True)
