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
from models import model_new_conv
from data import process
from data import data_input

# Hyper-parameters
LEARNING_RATE = 1e-6
TOTAL_EPOCH = 200
BATCH_SIZE = 100
DROPOUT_RATE = 0.1
RANDOM_SEED = int(np.random.random() * 1000)

CURRENT = time.time()

SESSION_NAME = '{}_lr{}_ep{}'.format(
    time.ctime(), LEARNING_RATE, TOTAL_EPOCH)
LOG_TRAIN_PATH = './log/' + SESSION_NAME + '/train/'
LOG_TEST_PATH = './log/' + SESSION_NAME + '/test/'
MODEL_PATH = LOG_TRAIN_PATH + 'model.ckpt'


def train(is_valid):
    # data set load
    x, y, angle = process.load_from_pickle()

    inputs = data_input.get_dataset(
        BATCH_SIZE, x, y, angle, is_shuffle=True, is_valid=is_valid)

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 75, 75, 9], name='X')
        Y = tf.placeholder(tf.float32, [None, 2], name='Y')
        A = tf.placeholder(tf.float32, [None, 5], name='A')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    _, xent, optimizer, accuracy = model_new_conv.make_model(
        X, Y, A, keep_prob, LEARNING_RATE)

    with tf.name_scope('hyperparam'):
        tf.summary.scalar('learning_rate', LEARNING_RATE)
        tf.summary.scalar('batch_size', BATCH_SIZE)
        tf.summary.scalar('dropout_rate', keep_prob)
        tf.summary.scalar('random_seed', RANDOM_SEED)

        print()
        print('Hyper Params')
        print("====================================================")
        print('Learning Rate', LEARNING_RATE)
        print('Batch Size', BATCH_SIZE)
        print('Dropout Rate', DROPOUT_RATE)
        print('Random Seed', RANDOM_SEED)
        print('\n')

    print('Train Start')

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        
        tf.global_variables_initializer().run()
        
        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, sess.graph)
        test_writer = tf.summary.FileWriter(LOG_TEST_PATH)

        total_batch = inputs.total_batch
        if is_valid:
            valid_total_batch = inputs.valid_total_batch

        probability = sess.graph.get_tensor_by_name('matrices/proba:0')

        for epoch in range(TOTAL_EPOCH):

            epoch_loss = epoch_acc = 0
            xs = ys = None

            for batch_num in range(total_batch):
                xs, ys, ans = inputs.next_batch(RANDOM_SEED, valid_set=False)

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
            if epoch % 20 == 19 or epoch == 0:
                print('[{:05.3f}] TRAIN EP: {:05d} | loss: {:0.5f} | acc: {:0.5f}'
                      .format(time.time() - CURRENT, epoch, epoch_loss, epoch_acc))

            # valid
            if is_valid:

                epoch_loss = epoch_acc = logloss = 0
                xs = ys = None

                for batch_num in range(valid_total_batch):
                    xs, ys, ans = inputs.next_batch(RANDOM_SEED, valid_set=True)

                    acc, loss, predict = sess.run([accuracy, xent, probability],
                                            feed_dict={X: xs, Y: ys, A: ans, keep_prob: 1})
                    epoch_loss += loss
                    epoch_acc += acc
                    logloss += log_loss(ys, predict)

                summary = sess.run(merged,
                                   feed_dict={X: xs, Y: ys, A: ans, keep_prob: 1})
                test_writer.add_summary(summary, epoch)

                epoch_loss = epoch_loss / total_batch
                epoch_acc = epoch_acc / valid_total_batch
                logloss = logloss / valid_total_batch
                if epoch % 20 == 19 or epoch == 0:
                    print('[{:05.3f}] VALID EP: {:05d} | loss: {:1.5f} | acc: {:1.5f} | logloss: {:1.5f}'
                            .format(time.time() - CURRENT, epoch, epoch_loss, epoch_acc, logloss))

        # model save
        saver.save(sess, MODEL_PATH)

    print('Train Finish')


if __name__ == '__main__':
    train(is_valid=True)
    # train(is_valid=False)