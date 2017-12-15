# Copy Right Kairos03 2017. All Right Reserved.

import tensorflow as tf

import model_sr
import model_conv
import data_input
import time

# Hyper-parameters
learning_rate = 1e-4
total_epoch = 30
batch_size = 30

name = '{}_lr{}_ep{}_b{}'.format(time.ctime(), learning_rate, total_epoch, batch_size)
log_path = './log/train/' + name

x, y = data_input.load_data()
inputs = data_input.get_dataset(batch_size, x, y)

saver = tf.train.Saver()


def train():
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 75, 75, 3])
        Y = tf.placeholder(tf.int32, [None, 2])

    model, xent, optimizer, accuracy = model_sr.make_model(X, Y, learning_rate)

    print('Train Start')
    with tf.Session() as sess:
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(log_path, sess.graph)
        tf.global_variables_initializer().run()

        total_batch = inputs.total_batch

        for epoch in range(total_epoch):

            epoch_loss = epoch_acc = 0
            xs = ys = None

            for batch_num in range(total_batch):
                xs, ys = inputs.next_batch()

                _, loss, acc = sess.run([optimizer, xent, accuracy],
                                        feed_dict={X: xs, Y: ys})
                epoch_loss += loss
                epoch_acc += acc

            summary = sess.run(merged, feed_dict={X: xs, Y: ys})
            train_writer.add_summary(summary, epoch)

            epoch_loss = epoch_loss / total_batch
            epoch_acc = epoch_acc / total_batch
            print('EP: {:05d}, loss: {:0.5f}, acc: {:0.5f}'.format(epoch, epoch_loss, epoch_acc))

        # model save
        saver.save(sess, log_path+'/model.ckpt')

    print('Train Finish')


def test():
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 75, 75, 3])
        Y = tf.placeholder(tf.int32, [None, 2])


if __name__ == '__main__':
    train()
