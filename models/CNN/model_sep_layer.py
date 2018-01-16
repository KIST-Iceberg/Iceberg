# Copy Right Kairos03 2017. All Right Reserved.
"""
"""

import tensorflow as tf


def var_summary(var):
    """ weight variable summary
    """
    with tf.name_scope('summary'):
        
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)

        tf.summary.histogram('histogram', var)


def single_stam(inputs, filters, name):

    c1_name = name+'_1'

    with tf.variable_scope(c1_name):
        layer = tf.layers.batch_normalization(inputs)
        layer = tf.layers.conv2d(layer, filters, (3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)
        print('[{:s}] \t | {}'.format(c1_name, layer.shape))

        # for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=c1_name):
        #     var_summary(var)
        var_summary(layer)
    
    with tf.variable_scope(name+'_maxpool'):
        layer = tf.layers.max_pooling2d(layer, (3, 3), strides=(2, 2), padding='same')

    return layer

def single_conv(inputs, filters, name):

    c1_name = name+'_1'
    c2_name = name+'_2'

    with tf.variable_scope(c1_name):
        layer = tf.layers.batch_normalization(inputs)
        layer = tf.layers.conv2d(layer, filters[0], (3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)
        print('[{:s}] \t | {}'.format(c1_name, layer.shape))
        
        # for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=c1_name):
        #     var_summary(var)
        var_summary(layer)

    with tf.variable_scope(c2_name):
        layer = tf.layers.batch_normalization(layer)
        layer = tf.layers.conv2d(layer, filters[1], (3, 3), strides=(1, 1), padding='same', activation=tf.nn.relu)
        print('[{:s}] \t | {}'.format(c2_name, layer.shape))

        # for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=c2_name):
        #     var_summary(var)
        var_summary(layer)
    
    with tf.variable_scope(name+'_maxpool'):
        layer = tf.layers.max_pooling2d(layer, (3, 3), strides=(2, 2), padding='same')

    return layer


def make_model(X, Y, A, keep_prob, learning_rate):

    with tf.variable_scope('hyperparam'):
        tf.summary.scalar('learning_rate', learning_rate)

    c1 = tf.reshape(X[:,:,:,0], (-1,75,75,1))
    c2 = tf.reshape(X[:,:,:,1], (-1,75,75,1))
    c3 = tf.reshape(X[:,:,:,2], (-1,75,75,1))

    with tf.name_scope('input_images'):
        tf.summary.image('band1', c1, 1)
        tf.summary.image('band2', c2, 1)
        tf.summary.image('mean_lee', c3, 1)

    c1 = single_stam(c1, filters=32, name='stam1_1')
    c2 = single_stam(c2, filters=32, name='stam1_2')
    c3 = single_stam(c3, filters=32, name='stam1_3')

    c1 = single_conv(c1, filters=(32, 64), name='conv1_1')
    c2 = single_conv(c2, filters=(32, 64), name='conv1_2')
    c3 = single_conv(c3, filters=(32, 64), name='conv1_3')

    c1 = single_conv(c1, filters=(128, 256), name='conv2_1')
    c2 = single_conv(c2, filters=(128, 256), name='conv2_2')
    c3 = single_conv(c3, filters=(128, 256), name='conv2_3')

    with tf.name_scope('image_concat'):
        imconcat = tf.concat([c1,c2,c3], axis=3)

        print('[{:s}] \t | {}'.format('img_concat', imconcat.shape))
        assert imconcat.shape[1:4] == (10,10,768)
    
    conv = single_conv(imconcat, filters=(712, 1024), name='imcat_conv')
    assert conv.shape[1:4] == (5,5,1024)

    with tf.name_scope('avg_pool'): 
        pool = tf.layers.average_pooling2d(conv, 5, 5, padding='same')

        print('[{:s}] \t | {}'.format('avg_pool', pool.shape))
        assert pool.shape[1:4] == (1,1,1024)

    with tf.name_scope('reshape_and_add_data'):
        reshape = tf.reshape(pool, (-1,1024))
        addition = tf.concat([reshape, A], axis=1)

        print('[{:s}] \t | {}'.format('add_data', addition.shape))
        assert addition.shape[-1] == (1029)
    
    with tf.variable_scope('dense1'):
        bn = tf.layers.batch_normalization(addition)
        dense = tf.layers.dense(bn, 512, activation=tf.nn.relu)
        drop = tf.layers.dropout(dense, rate=keep_prob)
        print('[{:s}] \t | {}'.format('dense1', drop.shape))

        # for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dense1'):
        #     var_summary(var)
        var_summary(dense)

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
