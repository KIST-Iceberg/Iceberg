#

import tensorflow as tf



def var_summary(var):
    """ weight variable summary
    """
    tf.summary.histogram('histogram', var)
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)


def make_conv(layer, filters, keep_prob, name):
    with tf.variable_scope(name):
        layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.conv2d(layer, filters, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.dropout(layer, rate=keep_prob)

        print('[{:s}] \t | \t {}'.format(name, layer.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name):
            var_summary(var)
        
    return layer


def make_max_pool(layer, name):
    with tf.variable_scope(name):
        layer = tf.layers.max_pooling2d(layer, (3, 3), strides=(2, 2))

        print('[{:s}] \t | \t {}'.format(name, layer.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name):
            var_summary(var)

    return layer


def make_dense(layer, hidden, keep_prob, name):
    with tf.variable_scope(name):
        layer = tf.layers.dense(layer, hidden, activation=tf.nn.relu)
        layer = tf.layers.dropout(layer, rate=keep_prob)

        print('[{:s}] \t | \t {}'.format(name, layer.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name):
            var_summary(var)

    return layer


def make_model(X, Y, A, learning_rate, keep_prob):

    layer = make_conv(X, 64, keep_prob, name="conv1")
    layer = make_max_pool(layer, name='maxpool1')

    layer = make_conv(layer, 128, keep_prob, name='conv2')
    layer = make_max_pool(layer, name='maxpool2')

    layer = make_conv(layer, 128, keep_prob, name='conv3')
    layer = make_max_pool(layer, name='maxpool3')

    layer = make_conv(layer, 64, keep_prob, name='conv4')
    layer = make_max_pool(layer, name='maxpool4')

    with tf.variable_scope('reshape'):
        layer = tf.reshape(layer, (-1, 3 * 3 * 64))
        print('[{:s}] \t | \t {}'.format('reshape', layer.shape))

    with tf.variable_scope('add_data'):
        layer = tf.concat((layer, A), axis=1)
        print('[{:s}] \t | \t {}'.format('add_data', layer.shape))

    layer = make_dense(layer, 512, keep_prob, name='dense1')

    output = tf.layers.dense(layer, 2)


    with tf.name_scope('matrices'):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output), name='xent')
        tf.summary.scalar('xent', xent)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(xent)

        correct = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

        tf.summary.scalar('accuracy', accuracy)
        proba = tf.nn.softmax(output, name='proba')

    return xent, optimizer, accuracy