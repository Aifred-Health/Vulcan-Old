"""Contains activation functions and gradient descent optimizers."""

import tensorflow as tf
from tf.nn import sigmoid, softmax, selu, relu
from tf.contrib.keras.optimizers import SGD, Adam

activations = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "rectify": relu,
    "selu": selu
}

optimizers = {
    "sgd": SGD,
    "adam": Adam
}
