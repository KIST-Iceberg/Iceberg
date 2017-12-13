# Copy Right Kairos03 2017. All Right Reserved.

import tensorflow as tf


def var_summary(var):
    tf.summary.histogram('histogram', var)
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)


def conv2d(inputs, filters, ksize, strides, padding, activation, keep_prob, name, is_deconv=False):
    """ make a deconv layer

        Args:
            inputs: inputs, 4-dim tensor
            filters: filters of layer, int
            ksize: kernel_size, 2-dim int list
            strides: 2-dim int list
            padding: 'SAME' or 'VALID'
            activation: activation function, if None skipped
            keep_prob: parameter of drop_out, float
            name: name of layer
            is_deconv: if True use tf.layers.conv2d_transpose else tf.layers.conv2d, default: False

        Return:
             layer: conv layer
    """

    # select layer method
    layer_method = tf.layers.conv2d
    if is_deconv:
        layer_method = tf.layers.conv2d_transpose

    # make layer
    with tf.name_scope(name):
        layer = layer_method(inputs, filters=filters, kernel_size=ksize, strides=strides, padding=padding)
        if activation is not None:
            layer = activation(layer)
        droped = tf.layers.dropout(layer, rate=keep_prob)

        var_summary(droped)

    return droped


def make_conv_layers(inputs, filters, ksizes, strieds, paddings, activations, keep_probs, is_deconv=False):
    """ Make conv Layers

        Args:
            inputs: inputs, 4-dim tensor
            filters: lists of filters, int list
            ksizes: lists of kernel_size or a list of kernel_size, 2-dim int list
            strieds: lists of strieds or a list of strides, 2-dim int list
            paddings: list of padding or a padding, 'SAME' or 'VALID'
            activations: list of activation or a activation, function
            keep_probs: list of keep_prob or a keep_prob, float
            is_deconv: if True use tf.layers.conv2d_transpose else tf.layers.conv2d, default: False

        Return:
            layer: super-resolution layer
    """

    layer_name = 'conv_'
    if is_deconv:
        layer_name = 'deconv_'

    next_inputs = inputs
    print('input: ' + str(next_inputs.shape))

    # make layers
    for layer_num in range(len(filters)):
        next_inputs = conv2d(next_inputs,
                             filters=filters[layer_num],
                             ksize=ksizes[layer_num],
                             strides=strieds[layer_num],
                             padding=paddings[layer_num],
                             activation=activations[layer_num],
                             keep_prob=keep_probs[layer_num],
                             name=layer_name + str(layer_num),
                             is_deconv=is_deconv)
        print("conv_" + str(layer_num)+': ' + str(next_inputs.shape))

    return next_inputs


def dense(inputs, out_dim, activation, keep_prob, name):
    """ dense layer
        Args:
            inputs: tensor
            out_dim: output dim, int
            activation: activation function
            keep_prob: drop_out parameter, float
            name: name of layer

        Return:
            layer: dense layer
    """
    with tf.name_scope(name):
        layer = tf.layers.dense(inputs, out_dim)
        if activation is not None:
            layer = activation(layer)
        droped = tf.layers.dropout(layer, keep_prob)

        var_summary(droped)

        return droped


def make_dense_layer(inputs, out_dims, activations, keep_probs):
    """ dense layer
            Args:
                inputs: tensor
                out_dims: output dim, list of int
                activations: list of activation function
                keep_probs: list of drop_out parameter, float

            Return:
                layer: dense layers
    """

    next_inputs = inputs
    print('input: ' + str(next_inputs.shape))

    for layer_num in range(len(out_dims)):
        next_inputs = dense(inputs,
                            out_dim=out_dims[layer_num],
                            activation=activations[layer_num],
                            keep_prob=keep_probs[layer_num],
                            name='dense_'+str(layer_num))
        print("dense_" + str(layer_num) + ': ' + str(next_inputs.shape))

    return next_inputs

