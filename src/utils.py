"""Contains auxilliary methods."""
import os

import numpy as np

import pickle

import matplotlib.pyplot as plt


def round_list(raw_list, decimals=4):
    """
    Return the same list with each item rounded off.

    Args:
        raw_list: float list
        decimals: how many decimal points to round to

    Returns: the rounded list
    """
    return [round(item, decimals) for item in raw_list]


def get_confusion_matrix(prediction, truth):
    """
    Calculate the confusion matrix for classification network predictions.

    Args:
        predicted: the class matrix predicted by the network.
                   Does not take one hot vectors.
        actual: the class matrix of the ground truth
                Does not take one hot vectors.

    Returns: the confusion matrix
    """
    from sklearn.metrics import confusion_matrix

    if len(prediction.shape) == 2:
        prediction = prediction[:, 0]
    if len(truth.shape) == 2:
        truth = truth[:, 0]

    return confusion_matrix(y_true=truth,
                            y_pred=prediction)


def get_one_hot(in_matrix):
    """
    Reformat truth matrix to same size as the output of the dense network.

    Args:
        in_matrix: the categorized 1D matrix

    Returns: a one-hot matrix representing the categorized matrix
    """
    if in_matrix.dtype.name == 'category':
        num_classes = len(in_matrix.cat.categories)
        custum_array = in_matrix.cat.codes

    elif isinstance(in_matrix, np.ndarray):
        num_classes = len(np.unique(in_matrix))
        custum_array = in_matrix

    else:
        print("Warning: Input matrix cannot be converted")
        num_classes = 0
        custum_array = []
        return

    temp = np.zeros(shape=(1, num_classes), dtype='float32')
    for i in np.array(custum_array):
        row = np.zeros((1, num_classes))
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


def display_record(record, load_path=None):
    """
    Display the training curve for a network training session.

    Args:
        record: the record dictionary for dynamic graphs during training
        load_path: the saved record .pickle file to load
    """
    if load_path is not None:
        with open(load_path) as in_file:
            record = pickle.load(in_file)
        plt.title('Training curve for model: %s' % os.path.basename(load_path))
    else:
        plt.title('Training curve')

    train_error, = plt.plot(
        record['epoch'],
        record['train_error'],
        '-mo',
        label='Train Error'
    )
    train_accuracy, = plt.plot(
        record['epoch'],
        record['train_accuracy'],
        '-go',
        label='Train Accuracy'
    )
    validation_error, = plt.plot(
        record['epoch'],
        record['validation_error'],
        '-ro',
        label='Validation Error'
    )
    validation_accuracy, = plt.plot(
        record['epoch'],
        record['validation_accuracy'],
        '-bo',
        label='Validation Accuracy'
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy error")
    # plt.ylim(0,1)

    plt.legend(handles=[train_error,
                        train_accuracy,
                        validation_error,
                        validation_accuracy],
               loc=0)

    plt.show()
    plt.pause(0.0001)
