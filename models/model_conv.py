# Copy Right Kairos03 2017. All Right Reserved.

import tensorflow as tf
import layers

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


# model
def make_model(X, Y, learning_rate):
    """ make model
    Args:
        X: input
        Y: label
        learning_rate: float
    Return:
        model: model
        xent: cross entropy
        optimizer: optimizer
        accuracy: accuracy
    """
    tf.summary.image('inputs', X, 4)

    with tf.name_scope('convolution'):
        cv_layers = layers.make_conv_layers(X,
                                            filters=cv_filters,
                                            ksizes=cv_ksizes,
                                            strieds=cv_strides,
                                            paddings=cv_paddings,
                                            activations=cv_activations,
                                            keep_probs=cv_keep_probs,
                                            is_deconv=False)

    with tf.name_scope('reshape'):
        reshaped = tf.reshape(cv_layers, rs_size)

    with tf.name_scope('dense'):
        ds_layers = layers.make_dense_layer(reshaped,
                                            out_dims=ds_out_dims,
                                            activations=ds_activations,
                                            keep_probs=ds_keep_probs)
    with tf.name_scope('output'):
        output = ds_layers

    with tf.name_scope('matrices'):
        with tf.name_scope('xent'):
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output))
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(xent)
        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return output, xent, optimizer, accuracy
