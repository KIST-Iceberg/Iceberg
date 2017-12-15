# Copy Right Kairos03 2017. All Right Reserved.

import tensorflow as tf

import model_sr
import data_input
import time

# Hyper-parameters
learning_rate = 3e-2
total_epoch = 10000
batch_size = 50

inputs = data_input.get_dataset(batch_size)


def var_summary(var):
    tf.summary.histogram('histogram', var)
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)


def train():
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 75, 75, 3])
        Y = tf.placeholder(tf.int32, [None, 2])

    model, xent, optimizer, accuracy = model_sr.make_model(X, Y, learning_rate)
    with tf.name_scope('matrices'):
        tf.summary.scalar('xent', xent)
        tf.summary.scalar('accuracy', accuracy)

    # vars_super_resolution = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='super_resolution')
    # vars_convolution = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='convolution')
    # vars_dense = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense')
    #
    # for vars in [vars_super_resolution, vars_convolution, vars_dense]:
    #     for var in vars:
    #         var_summary(var)

    print('Start')
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        name = 'lr{}_ep{}_b{}_{}'.format(learning_rate, total_epoch, batch_size, time.ctime())
        train_writer = tf.summary.FileWriter('./log/train/{}'.format(name), sess.graph)
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

    print('Finish')


if __name__ == '__main__':
    train()
