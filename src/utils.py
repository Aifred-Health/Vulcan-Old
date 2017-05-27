"""Contains auxilliary methods."""
import os

import numpy as np

import pickle

import matplotlib.pyplot as plt


def get_one_hot(in_matrix):
    """
    Reformat truth matrix to same size as the output of the dense network.

    Args:
        in_matrix: the categorized 1D matrix (dtype needs to be category)

    Returns: a one-hot matrix representing the categorized matrix
    """
    temp = np.zeros(shape=(1, len(in_matrix.cat.categories)), dtype='float32')
    for i in np.array(in_matrix.cat.codes):
        row = np.zeros((1, len(in_matrix.cat.categories)))
        row[0, i] = 1.0
        temp = np.concatenate((temp, row), axis=0)
    return np.array(temp[1:], dtype='float32')


def get_class(in_matrix):
    """
    Reformat truth matrix to be the classes in a 1D array.

    Args:
        in_matrix: one-hot matrix

    Returns: Class array
    """
    return np.expand_dims(np.argmax(in_matrix, axis=1), axis=1)


def display_record(load_path):
    """
    Display the training curve for a network training session.

    Args:
        load_path: the saved record .pickle file to load
    """
    with open(load_path) as in_file:
        record = pickle.load(in_file)

    plt.plot(
        record['epoch'],
        record['train_error'],
        '-mo',
        label='Train Error'
    )
    plt.plot(
        record['epoch'],
        record['train_accuracy'],
        '-go',
        label='Train Accuracy'
    )
    plt.plot(
        record['epoch'],
        record['validation_error'],
        '-ro',
        label='Validation Error'
    )
    plt.plot(
        record['epoch'],
        record['validation_accuracy'],
        '-bo',
        label='Validation Accuracy'
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy error")
    # plt.ylim(0,1)
    plt.title('Training curve for model: %s' % os.path.basename(load_path))
    plt.legend(loc='upper right')

    plt.show()
