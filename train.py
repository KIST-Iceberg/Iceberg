# Copy Right Kairos03 2017. All Right Reserved.

import tensorflow as tf

import model_sr
import data_input

# Hyper-parameters
learning_rate = 1e-4
total_epoch = 1000
batch_size = 100

inputs = data_input.get_dataset(batch_size)


def train():

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 75, 75, 3])
        Y = tf.placeholder(tf.float32, [None])

    model, xent, optimizer, accuracy = model_sr.make_model(X, Y, learning_rate)

    print('Start')
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch in range(total_epoch):

            epoch_loss = epoch_acc = 0

            for batch_num in range(inputs.total_batch):

                xs, ys = inputs.next_batch()

                _, loss, acc = sess.run([optimizer, xent, accuracy],
                                        feed_dict={X: xs, Y: ys})

                epoch_loss += loss
                epoch_acc += accuracy

            epoch_loss /= inputs.total_batch
            epoch_acc /= inputs.total_batch

            print('EP: {:05d}, loss: {:.5f}, acc: {:.5f}'.format(epoch, epoch_loss, epoch_acc))

    print('Finish')
