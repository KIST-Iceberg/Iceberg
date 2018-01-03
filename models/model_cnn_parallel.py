"""
Copyright kairos03. All Right Reserved.

Multi cell cnn model

@ this model is too complex to presenting Iceberg dataset so always overfitted.
"""
import tensorflow as tf
import layers


def make_model(X, Y, angle, keep_prob, learning_rate):

    with tf.name_scope('conv'):

        # conv
        conv1 = layers.conv_cell(X,
                                 filters=128,
                                 strides=[2, 2],
                                 keep_prob=keep_prob)
        print('conv1', conv1.shape)

        conv2 = layers.conv_cell(conv1,
                                 filters=512,
                                 strides=[2, 2],
                                 keep_prob=keep_prob)
        print('conv2', conv2.shape)

        max_pool1 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        print('max_pool1', max_pool1.shape)

        conv3 = layers.conv_cell(max_pool1,
                                 filters=512,
                                 strides=[2, 2],
                                 keep_prob=keep_prob)
        print('conv3', conv3.shape)

        conv4 = layers.conv_cell(conv3,
                                 filters=256,
                                 strides=[2, 2],
                                 keep_prob=keep_prob)
        print('conv4', conv4.shape)

        max_pool2 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        print('max_pool2', max_pool2.shape)

    with tf.name_scope('reshape_and_add_angle'):
        reshape = tf.reshape(max_pool2, [-1, 2 * 2 * 768])
        print('reshape', reshape.shape)
        print('angle', angle.shape)
        add_angel = tf.concat([reshape, angle], 1)
        print('add_angel', add_angel.shape)

    with tf.name_scope('dense'):
        dense1 = layers.dense(add_angel, 1024, tf.nn.relu, keep_prob, name='dense')
        print('dense1', dense1.shape)
        dense2 = layers.dense(dense1, 256, tf.nn.relu, keep_prob, name='dense')
        print('dense2', dense2.shape)
        dense3 = layers.dense(dense2, 2, tf.nn.relu, keep_prob, name='dense')
        print('dense3', dense3.shape)

    with tf.name_scope('output'):
        output = tf.identity(dense3, name='output')

    with tf.name_scope('matrices'):
        with tf.name_scope('xent'):
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output), name='xent')
            tf.summary.scalar('xent', xent)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(xent)
        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', accuracy)
        with tf.name_scope('proba'):
            proba = tf.nn.softmax(output, name='proba')

    return output, xent, optimizer, accuracy
