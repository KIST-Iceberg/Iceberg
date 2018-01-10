# Copy Right Kairos03 2017. All Right Reserved.
"""
"""

import tensorflow as tf


def var_summary(var):
    """ weight variable summary
    """
    with tf.name_scope('summary'):
        tf.summary.histogram('histogram', var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        

def make_model(X, Y, A, keep_prob, learning_rate):
    with tf.name_scope('input_images'):
        tf.summary.image('origin', X[:, :, :, 0:3], 1)
        tf.summary.image('lee', X[:, :, :, 3:6], 1)
        tf.summary.image('high', X[:, :, :, 6:9], 1)

    with tf.variable_scope('hyperparam'):
        tf.summary.scalar('learning_rate', learning_rate)

    with tf.variable_scope('conv1'):
        bn = tf.layers.batch_normalization(X)
        conv = tf.layers.conv2d(
            bn, 64, (3, 3), strides=(2, 2), activation=tf.nn.relu)
        drop = tf.layers.dropout(conv, keep_prob)
        print('[{:s}] \t | {}'.format('conv1', drop.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv1'):
            var_summary(var)

    with tf.variable_scope('conv2'):
        bn = tf.layers.batch_normalization(drop)
        conv = tf.layers.conv2d(
            bn, 128, (3, 3), strides=(2, 2), activation=tf.nn.relu)
        drop = tf.layers.dropout(conv, keep_prob)
        print('[{:s}] \t | {}'.format('conv2', drop.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv2'):
            var_summary(var)

    with tf.variable_scope('conv3'):
        bn = tf.layers.batch_normalization(drop)
        conv = tf.layers.conv2d(
            bn, 128, (3, 3), strides=(2, 2), activation=tf.nn.relu)
        drop = tf.layers.dropout(conv, keep_prob)
        print('[{:s}] \t | {}'.format('conv3', drop.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv3'):
            var_summary(var)

    with tf.variable_scope('conv4'):
        bn = tf.layers.batch_normalization(drop)
        conv = tf.layers.conv2d(
            bn, 64, (3, 3), strides=(2, 2), activation=tf.nn.relu)
        drop = tf.layers.dropout(conv, keep_prob)
        print('[{:s}] \t | {}'.format('conv4', drop.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv4'):
            var_summary(var)

    with tf.variable_scope('reshape'):
        layer = tf.reshape(drop, (-1, 3 * 3 * 64))
        print('[{:s}] \t | {}'.format('reshape', layer.shape))

    with tf.variable_scope('add_data'):
        layer = tf.concat((layer, A), axis=1)
        print('[{:s}] \t | {}'.format('add_data', layer.shape))

    with tf.variable_scope('dense1'):
        bn = tf.layers.batch_normalization(layer)
        dense = tf.layers.dense(bn, 1024, activation=tf.nn.relu)
        drop = tf.layers.dropout(dense, keep_prob)
        print('[{:s}] \t | {}'.format('dense1', drop.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense1'):
            var_summary(var)

    with tf.variable_scope('dense2'):
        bn = tf.layers.batch_normalization(drop)
        dense = tf.layers.dense(bn, 512, activation=tf.nn.relu)
        drop = tf.layers.dropout(dense, keep_prob)
        print('[{:s}] \t | {}'.format('dense2', drop.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense2'):
            var_summary(var)

    output = tf.layers.dense(drop, 2, name='output')
    print('[{:s}] \t | {}'.format('output', output.shape))

    with tf.name_scope('matrices'):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=Y, logits=output), name='xent')
        tf.summary.scalar('xent', xent)

        optimizer = tf.train.AdamOptimizer(
            learning_rate, name='optimizer').minimize(xent)

        correct = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct, tf.float32), name='accuracy')

        tf.summary.scalar('accuracy', accuracy)
        proba = tf.nn.softmax(output, name='proba')
    

    return output, xent, optimizer, accuracy
