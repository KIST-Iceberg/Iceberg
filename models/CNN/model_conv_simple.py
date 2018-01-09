# Copy Right Kairos03 2017. All Right Reserved.
"""
simple cnn
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


def make_conv(layer, filter_size, keep_prob, name):
    """
    make conv2d layer using preset
    """
    with tf.name_scope(name):
        conv = tf.layers.conv2d(layer, filter_size, (3, 3), padding='same')
        drop = tf.layers.dropout(conv, rate=keep_prob)

        print('[{:s}] \t | {}'.format(name, drop.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            var_summary(var)

    return drop


def make_max_pool(layer, name):
    """
    make max pool2d layer using preset
    """
    with tf.name_scope(name):
        layer = tf.layers.average_pooling2d(
            layer, (3, 3), strides=(2, 2), padding='same')

        print('[{:s}] \t | {}'.format(name, layer.shape))

    return layer


def make_dense(layer, hidden, keep_prob, name):
    """
    make dense layer using preset
    """
    with tf.name_scope(name):
        weight = tf.Variable(tf.truncated_normal(
            (int(layer.shape[1]), hidden), stddev=1.0))
        bias = tf.Variable(tf.constant(0.1, shape=[hidden]))
        dense = tf.matmul(layer, weight) + bias
        activation = tf.nn.relu(dense)
        drop = tf.layers.dropout(activation, rate=keep_prob)

        print('[{:s}] \t | {}'.format(name, drop.shape))

        var_summary(weight)
        var_summary(bias)

    return drop


def make_model(X, Y, A, learning_rate, keep_prob):
    """
    make model and return ops
    """

    print("Layer name \t | Shape")
    print("====================================================")

    tf.summary.image('input', X)

    layer = make_conv(X, 64, keep_prob, name="conv1")
    layer = make_max_pool(layer, name='maxpool1')

    layer = make_conv(layer, 128, keep_prob, name='conv2')
    layer = make_max_pool(layer, name='maxpool2')

    layer = make_conv(layer, 128, keep_prob, name='conv3')
    layer = make_max_pool(layer, name='maxpool3')

    layer = make_conv(layer, 64, keep_prob, name='conv4')
    layer = make_max_pool(layer, name='maxpool4')

    with tf.variable_scope('reshape'):
        layer = tf.reshape(layer, (-1, 5 * 5 * 64))
        print('[{:s}] \t | {}'.format('reshape', layer.shape))

    with tf.variable_scope('add_data'):
        layer = tf.concat((layer, A), axis=1)
        print('[{:s}] \t | {}'.format('add_data', layer.shape))

    layer = make_dense(layer, 512, keep_prob, name='dense1')

    output = tf.layers.dense(layer, 2, name='output')
    print('[{:s}] \t | {}'.format('add_data', output.shape))
    print("====================================================")

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
