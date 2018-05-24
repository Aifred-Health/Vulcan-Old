"""Contains activation functions and gradient descent optimizers."""

from lasagne.nonlinearities import sigmoid, softmax, rectify
from lasagne.updates import sgd, adam
from selu import selu

activations = {
    "sigmoid": sigmoid,
    "softmax": softmax,
    "rectify": rectify,
    "selu": selu
}

optimizers = {
    "sgd": sgd,
    "adam": adam
}
