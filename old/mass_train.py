# Copy Right Kairos03 2017. All Right Reserved.

import tensorflow as tf

from data import data_input
import time

# Hyper-parameters
learning_rate = 1e-4
total_epoch = 1000
batch_size = 50

# model parameter
# conv layer
# 75 75 36 18 9 3
cv_filters = [64, 256, 1024, 512, 256]
cv_ksizes = [[3, 3], [4, 4], [3, 3], [5, 5], [4, 4]]
cv_strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]]
cv_paddings = ['SAME', 'VALID', 'SAME', 'SAME', 'VALID']
cv_activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]
cv_keep_probs = [0.9, 0.9, 0.9, 0.9, 0.9]

# reshape
rs_size = [-1, 3 * 3 * 256]

# dense layer
ds_out_dims = [1024, 512, 2]
ds_activations = [tf.nn.relu, tf.nn.relu, None]
ds_keep_probs = [0.9, 0.9, 0.9]

# data
x, y = data_input.load_data()
inputs = data_input.get_dataset(batch_size, x, y)


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

        layer = tf.layers.batch_normalization(X)

    with tf.name_scope('conv0'):
        layer = tf.layers.conv2d(layer,
                                 filters=cv_filters[0],
                                 kernel_size=cv_ksizes[0],
                                 strides=cv_strides[0],
                                 padding=cv_paddings[0],
                                 activation=cv_activations[0])
        layer = tf.layers.dropout(layer, cv_keep_probs[0])

    with tf.name_scope('conv1'):
        layer = tf.layers.conv2d(layer,
                                 filters=cv_filters[1],
                                 kernel_size=cv_ksizes[1],
                                 strides=cv_strides[1],
                                 padding=cv_paddings[1],
                                 activation=cv_activations[1])
        layer = tf.layers.dropout(layer, cv_keep_probs[1])

    with tf.name_scope('conv2'):
        layer = tf.layers.conv2d(layer,
                                 filters=cv_filters[2],
                                 kernel_size=cv_ksizes[2],
                                 strides=cv_strides[2],
                                 padding=cv_paddings[2],
                                 activation=cv_activations[2])
        layer = tf.layers.dropout(layer, cv_keep_probs[2])

    with tf.name_scope('conv3'):
        layer = tf.layers.conv2d(layer,
                                 filters=cv_filters[3],
                                 kernel_size=cv_ksizes[3],
                                 strides=cv_strides[3],
                                 padding=cv_paddings[3],
                                 activation=cv_activations[3])
        layer = tf.layers.dropout(layer, cv_keep_probs[3])

    with tf.name_scope('conv4'):
        layer = tf.layers.conv2d(layer,
                                 filters=cv_filters[4],
                                 kernel_size=cv_ksizes[4],
                                 strides=cv_strides[4],
                                 padding=cv_paddings[4],
                                 activation=cv_activations[4])
        layer = tf.layers.dropout(layer, cv_keep_probs[4])

    with tf.name_scope('reshape'):
        reshaped = tf.reshape(layer, rs_size)

    with tf.name_scope('dense0'):
        layer = tf.layers.dense(reshaped,
                                ds_out_dims[0],
                                activation=ds_activations[0])

    with tf.name_scope('dense1'):
        layer = tf.layers.dense(layer,
                                ds_out_dims[1],
                                activation=ds_activations[1])

    with tf.name_scope('dense2'):
        layer = tf.layers.dense(layer,
                                ds_out_dims[2],
                                activation=ds_activations[2])

    with tf.name_scope('output'):
        output = layer

    with tf.name_scope('matrices'):
        with tf.name_scope('xent'):
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output))
            tf.summary.scalar('xent', xent)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(xent)
        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

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
