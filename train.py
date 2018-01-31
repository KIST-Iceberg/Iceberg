# Copy Right Kairos03 2017. All Right Reserved.
"""
train
"""
from __future__ import absolute_import

import time

import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss

from models.CNN import model_sep_layer
from data import process
from data import data_input

# Hyper-parameters
TOTAL_EPOCH = 800
BATCH_SIZE = 100
STARTER_LEARNING_RATE = 1e-5
DECAY_RATE = 0.8
DECAY_STEPS = 50
DROPOUT_RATE = 0.8
RANDOM_SEED = int(np.random.random() * 1000)
#RANDOM_SEED = 981

CURRENT = time.time()

SESSION_NAME = '{}_lr{}_ep{}'.format(
    time.ctime(), STARTER_LEARNING_RATE, TOTAL_EPOCH)
LOG_TRAIN_PATH = './log/' + SESSION_NAME + '/train/'
LOG_TEST_PATH = './log/' + SESSION_NAME + '/test/'
MODEL_PATH = LOG_TRAIN_PATH + 'model.ckpt'


def decayed_learning_rate(learning_rate, decay_rate, global_step, decay_steps):
    return learning_rate * pow(decay_rate, (global_step // decay_steps))


def train(is_valid):
    """
    train phase
    """
    # data set load
    x, y, angle = process.load_from_pickle()

    # make dataset
    inputs = data_input.get_dataset(
        BATCH_SIZE, x, y, angle, is_shuffle=True, is_valid=is_valid)

    # input placeholder
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 75, 75, 3], name='X')
        Y = tf.placeholder(tf.float32, [None, 2], name='Y')
        A = tf.placeholder(tf.float32, [None, 5], name='A')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # get train ops
    _, xent, optimizer, accuracy = model_sep_layer.make_model(
        X, Y, A, keep_prob, learning_rate)

    # summary hyperparam
    with tf.variable_scope('hyperparam'):
        tf.summary.scalar('batch_size', BATCH_SIZE)
        tf.summary.scalar('dropout_rate', keep_prob)
        tf.summary.scalar('random_seed', RANDOM_SEED)
        tf.summary.scalar('decay_rate', DECAY_RATE)
        tf.summary.scalar('decay_steps', DECAY_STEPS)

        print()
        print('Hyper Params')
        print("====================================================")
        print('Start Learning Rate', STARTER_LEARNING_RATE)
        print('Batch Size', BATCH_SIZE)
        print('Dropout Rate', DROPOUT_RATE)
        print('Random Seed', RANDOM_SEED)
        print('\n')

    # train start
    print('Train Start')

    # define saver and summary marge
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    # start session
    with tf.Session() as sess:

        # initalize global variables
        tf.global_variables_initializer().run()

        # define summary writer and write graph
        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, sess.graph)
        test_writer = tf.summary.FileWriter(LOG_TEST_PATH)

        # total_batch
        total_batch = inputs.total_batch
        if is_valid:
            valid_total_batch = inputs.valid_total_batch

        # get probabiliy op
        probability = sess.graph.get_tensor_by_name('matrices/proba:0')

        for epoch in range(TOTAL_EPOCH):
            
            # initialize epoch variable
            epoch_loss = epoch_acc = 0
            xs = ys = None
            decayed_lr = decayed_learning_rate(STARTER_LEARNING_RATE, DECAY_RATE, epoch, DECAY_STEPS)

            # start batch
            for batch_num in range(total_batch):
                
                # get data
                xs, ys, ans = inputs.next_batch(RANDOM_SEED, valid_set=False)

                # run ops
                _, loss, acc = sess.run([optimizer, xent, accuracy],
                                        feed_dict={X: xs, Y: ys, A: ans, keep_prob: DROPOUT_RATE,
                                                   learning_rate: decayed_lr})
                
                # sum batch loss and accuracy
                epoch_loss += loss
                epoch_acc += acc

                # display time spend for 20 batch
                # if batch_num % 20 == 0:
                #     print("[{:05.3f}] Batch {} finish.".format(time.time() - last_time, batch_num))
                #     last_time = time.time()

            # write train summary
            summary = sess.run(merged,
                               feed_dict={X: xs, Y: ys, A: ans, keep_prob: DROPOUT_RATE,
                                          learning_rate: decayed_lr})
            train_writer.add_summary(summary, epoch)

            # calculate epoch loss, acurracy and display its
            epoch_loss = epoch_loss / total_batch
            epoch_acc = epoch_acc / total_batch
            if epoch % 20 == 19 or epoch == 0:
                print('[{:05.3f}] TRAIN EP: {:05d} | loss: {:0.5f} | acc: {:0.5f}'
                      .format(time.time() - CURRENT, epoch, epoch_loss, epoch_acc))

            # valid 
            if is_valid:
                
                # initialize epoch variables
                epoch_loss = epoch_acc = logloss = 0
                xs = ys = None

                # start batch
                for batch_num in range(valid_total_batch):
                    xs, ys, ans = inputs.next_batch(
                        RANDOM_SEED, valid_set=True)

                    # run ops
                    acc, loss, predict = sess.run([accuracy, xent, probability],
                                                  feed_dict={X: xs, Y: ys, A: ans, keep_prob: 1,
                                                             learning_rate: decayed_lr})
                    # sum batch loss, accuracy, logloss
                    epoch_loss += loss
                    epoch_acc += acc
                    logloss += log_loss(ys, predict)

                # write test summary
                summary = sess.run(merged,
                                   feed_dict={X: xs, Y: ys, A: ans, keep_prob: 1,
                                              learning_rate: decayed_lr})
                test_writer.add_summary(summary, epoch)

                # calculate epoch loss, accuracy, logloss and display its
                epoch_loss = epoch_loss / total_batch
                epoch_acc = epoch_acc / valid_total_batch
                logloss = logloss / valid_total_batch
                if epoch % 20 == 19 or epoch == 0:
                    print('[{:05.3f}] VALID EP: {:05d} | loss: {:1.5f} | acc: {:1.5f} | logloss: {:1.5f}'
                          .format(time.time() - CURRENT, epoch, epoch_loss, epoch_acc, logloss))

        # model save
        saver.save(sess, MODEL_PATH)

    del inputs, x, y, angle

    print('Train Finish')


if __name__ == '__main__':
    train(is_valid=True)
    # train(is_valid=False)