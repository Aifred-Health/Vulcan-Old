"""Contains activation functions and gradient descent optimizers."""

import tensorflow as tf


activations = {
    "sigmoid": tf.nn.sigmoid,
    "softmax": tf.nn.softmax,
    "rectify": tf.nn.relu,
    "selu": tf.nn.selu
}

optimizers = {
    "sgd": tf.train.GradientDescentOptimizer,
    "adam": tf.train.AdamOptimizer
}
