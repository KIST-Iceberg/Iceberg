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

def make_avg_pool(layer, name):
    """
    make avg pool2d layer using preset
    """
    with tf.name_scope(name):
        layer = tf.layers.average_pooling2d(
            layer, (3, 3), strides=(2, 2), padding='same')

        print('[{:s}] \t | {}'.format(name, layer.shape))

    return layer


def make_model(X, Y, A, keep_prob, learning_rate):

    tf.summary.image('origin', X[:,:,:,0:3])
    tf.summary.image('lee', X[:,:,:,3:6])
    tf.summary.image('high', X[:,:,:,6:9])

    with tf.variable_scope('conv1'):
        bn = tf.layers.batch_normalization(X)
        conv = tf.layers.conv2d(bn, 64, (3, 3),strides=(2, 2), activation=tf.nn.relu, padding='SAME')
        drop = tf.layers.dropout(conv, rate = keep_prob)
        maxpool = tf.layers.max_pooling2d(drop, (3,3), strides=(2,2), padding='SAME', name='maxpool')
        print('[{:s}] \t | {}'.format('conv1', avg.shape))
        
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv1'):
            var_summary(var)

    with tf.variable_scope('conv2'):
        bn = tf.layers.batch_normalization(maxpool)
        conv = tf.layers.conv2d(bn, 128, (3, 3),strides=(2, 2), activation=tf.nn.relu, padding='SAME')
        drop = tf.layers.dropout(conv,  rate = keep_prob)
        avgpool1 = make_avg_pool(drop, name='avgpool1')
        print('[{:s}] \t | {}'.format('conv2', avgpool1.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv2'):
            var_summary(var)

    with tf.variable_scope('conv3'):
        bn = tf.layers.batch_normalization(avgpool1)
        conv = tf.layers.conv2d(bn, 64, (3, 3),strides=(2, 2), activation=tf.nn.relu, padding='SAME')
        drop = tf.layers.dropout(conv,  rate = keep_prob)
        # avgpool2 = make_avg_pool(drop, name='avgpool2')
        print('[{:s}] \t | {}'.format('conv3', drop.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv3'):
            var_summary(var)
    '''
    with tf.variable_scope('conv4'):
        bn = tf.layers.batch_normalization(drop)
        conv = tf.layers.conv2d(bn, 64, (3, 3),strides=(2, 2), activation=tf.nn.relu)
        drop = tf.layers.dropout(conv)
        avgpool4 = make_avg_pool(drop, name='avgpool4')
        print('[{:s}] \t | {}'.format('conv4', avgpool4.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv4'):
            var_summary(var)
    '''
    with tf.variable_scope('reshape'):
        layer = tf.reshape(drop, (-1, 3 * 3 * 64))
        print('[{:s}] \t | {}'.format('reshape', layer.shape))

    with tf.variable_scope('add_data'):
        layer = tf.concat((layer, A), axis=1)
        print('[{:s}] \t | {}'.format('add_data', layer.shape))
    
    with tf.variable_scope('dense1'):
        bn = tf.layers.batch_normalization(layer)
        dense = tf.layers.dense(bn, 1024, activation=tf.nn.relu)
        drop = tf.layers.dropout(dense)
        print('[{:s}] \t | {}'.format('dense1', drop.shape))

        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense1'):
            var_summary(var)
    
    with tf.variable_scope('dense2'):
        bn = tf.layers.batch_normalization(drop)
        dense = tf.layers.dense(bn, 512, activation=tf.nn.relu)
        drop = tf.layers.dropout(dense)
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
