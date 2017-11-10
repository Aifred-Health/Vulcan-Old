"""Contains activation functions and gradient descent optimizers."""
from lasagne.nonlinearities import sigmoid, softmax, rectify
from selu import selu
from lasagne.updates import sgd, adam

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
