# Copy Right Kairos03 2017. All Right Reserved.

import tensorflow as tf


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
            act = activation(layer)
        droped = tf.layers.dropout(act, rate=keep_prob)

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

    next_inputs = inputs

    # make layers
    for layer_num in range(len(filters)):
        next_inputs = conv2d(next_inputs,
                             filters=filters[layer_num],
                             ksize=ksizes[layer_num],
                             strides=strieds[layer_num],
                             padding=paddings[layer_num],
                             activation=activations[layer_num],
                             keep_prob=keep_probs[layer_num],
                             name='deconv_' + str(layer_num),
                             is_deconv=is_deconv)

    return next_inputs

