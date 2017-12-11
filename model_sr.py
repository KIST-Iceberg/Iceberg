# Copy Right Kairos03 2017. All Right Reserved.

import tensorflow as tf
import layers

# model parameter
# super resolution
sr_filters = [64, 32, 3]
sr_ksizes = [[9, 9], [5, 5], [5, 5]]
sr_strides = [[2, 2], [2, 2], [2, 2]]
sr_paddings = ['SAME', 'SAME', 'SAME']
sr_activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu]
sr_keep_probs = [0.9, 0.9, 0.9]

# conv layer
# 75 36 18 9 5
cv_filters = [64, 128, 128, 64]
cv_ksizes = [[4, 4], [3, 3], [3, 3], [2, 2]]
cv_strides = [[2, 2], [2, 2], [2, 2], [2, 2]]
cv_paddings = ['VALID', 'SAME', 'SAME', 'VALID']
cv_activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]
cv_keep_probs = [0.9, 0.9, 0.9, 0.9]

# reshape
rs_size = [-1, 5 * 5 * 64]

# dense layer
ds_out_dims = [512, 256, 1]
ds_activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu]
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
    with tf.name_scope('super_resolution'):
        sr_layers = layers.make_conv_layers(X,
                                            filters=sr_filters,
                                            ksizes=sr_ksizes,
                                            strieds=sr_strides,
                                            paddings=sr_paddings,
                                            activations=sr_activations,
                                            keep_probs=sr_keep_probs,
                                            is_deconv=True)

    with tf.name_scope('convolution'):
        cv_layers = layers.make_conv_layers(sr_layers,
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

    with tf.name_scope('matrices'):
        with tf.name_scope('xent'):
            xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=ds_layers))
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(xent)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.case(tf.equal(Y, ds_layers), tf.float32))

    return ds_layers, xent, optimizer, accuracy