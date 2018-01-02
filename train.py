# Copy Right Kairos03 2017. All Right Reserved.

import tensorflow as tf

from models import model_cnn_parallel
from data import data_input
import time

# Hyper-parameters
learning_rate = 1e-4
total_epoch = 30
batch_size = 30
dropout_kp = 0.2

current = time.time()

name = '{}_lr{}_ep{}_b{}'.format(time.ctime(), learning_rate, total_epoch, batch_size)
log_train_path = './log/' + name + '/train/'
log_test_path = './log/' + name + '/test/'
model_path = log_train_path + 'model.ckpt'


def train(is_valid):
    # data set load
    x, y, idx = data_input.load_data()
    inputs = data_input.get_dataset(batch_size, x, y, is_shuffle=True, is_valid=is_valid)

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 75, 75, 3], name='X')
        Y = tf.placeholder(tf.float32, [None, 2], name='Y')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    model, xent, optimizer, accuracy = model_cnn_parallel.make_model(X, Y, keep_prob, learning_rate)

    print('Train Start')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(log_train_path, sess.graph)
        test_writer = tf.summary.FileWriter(log_test_path)
        tf.global_variables_initializer().run()

        total_batch = inputs.total_batch
        valid_total_batch = inputs.valid_total_batch

        for epoch in range(total_epoch):

            epoch_loss = epoch_acc = 0
            xs = ys = None

            for batch_num in range(total_batch):
                xs, ys = inputs.next_batch()

                tf.summary.image('input', xs)

                _, loss, acc = sess.run([optimizer, xent, accuracy],
                                        feed_dict={X: xs, Y: ys, keep_prob: dropout_kp})
                epoch_loss += loss
                epoch_acc += acc

            summary = sess.run(merged, feed_dict={X: xs, Y: ys, keep_prob: dropout_kp})
            train_writer.add_summary(summary, epoch)

            epoch_loss = epoch_loss / total_batch
            epoch_acc = epoch_acc / total_batch
            print('[{:05.3f}] EP: {:05d}, loss: {:0.5f}, acc: {:0.5f}'
                  .format(time.time() - current, epoch, epoch_loss, epoch_acc))

            # valid
            if is_valid:

                epoch_acc = 0
                xs = ys = None

                for batch_num in range(valid_total_batch):
                    xs, ys = inputs.next_batch(valid_set=True)

                    acc = sess.run(accuracy,
                                   feed_dict={X: xs, Y: ys, keep_prob: 1})

                    epoch_acc += acc

                summary = sess.run(merged, feed_dict={X: xs, Y: ys, keep_prob: 1})
                test_writer.add_summary(summary, epoch)

                epoch_acc = epoch_acc / valid_total_batch
                print('[{:05.3f}] VALID EP: {:05d}, acc: {:0.5f}'
                      .format(time.time() - current, epoch, epoch_acc))

        # model save
        saver.save(sess, model_path)

    print('Train Finish')


if __name__ == '__main__':
    train(is_valid=True)
