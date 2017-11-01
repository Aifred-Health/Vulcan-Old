import os
import urllib
import gzip
import numpy as np
from sklearn.model_selection import train_test_split


def load_fashion_mnist(validation_size=None, validation_seed=None):
    """
    Get the fashion MNIST training data (downloading it if it is not already accessible),
    and return it as NumPy arrays.

    Extracted from https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md

    Args:
        validation_size: If not None or zero, set this much data aside for a validation set.
            Floats in (0,1) are interpreted as proportions (e.g., 0.1 uses 10% of the data for validation)
            Integers are interpreted as an absolute number of samples (e.g., 5000 puts exactly 5000 samples in the set)
        validation_seed: Seed for shuffling/selecting validation set

    :return: (train_images, train_labels, test_images, test_labels) if validation_proportion=None or zero, otherwise
        returns (train_images, train_labels, test_images, test_labels, val_images, val_labels)
    """

    if os.path.exists("data/fashion"):
        print("data folder already exists")
    else:
        print("Creating data/fashion folder")
        os.makedirs("data/fashion")

    if not os.path.exists("data/fashion/train-images-idx3-ubyte.gz"):
        print("No fashion MNIST training images found--downloading")
        _download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                       'data/fashion')
    train_images = _load_image('data/fashion/train-images-idx3-ubyte.gz')

    if not os.path.exists("data/fashion/train-labels-idx1-ubyte.gz"):
        print("No fashion MNIST training labels found--downloading")
        _download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
                       'data/fashion')
    train_labels = _load_label('data/fashion/train-labels-idx1-ubyte.gz')

    if not os.path.exists("data/fashion/t10k-images-idx3-ubyte.gz"):
        print("No fashion MNIST test (10k) images found--downloading")
        _download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
                       'data/fashion')
    test_images = _load_image("data/fashion/t10k-images-idx3-ubyte.gz")

    if not os.path.exists("data/fashion/t10k-labels-idx1-ubyte.gz"):
        print("No fashion MNIST test (10k) labels found--downloading")
        _download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
                       'data/fashion')
    test_labels = _load_label('data/fashion/t10k-labels-idx1-ubyte.gz')

    if validation_size is None or validation_size == 0:
        return train_images, train_labels, test_images, test_labels
    else:
        (train_images, val_images,  train_labels, val_labels) = train_test_split(train_images, train_labels,
                                                                                 random_state=validation_seed,
                                                                                 test_size=validation_size,
                                                                                 stratify=train_labels)
        return train_images, train_labels, test_images, test_labels, val_images, val_labels


def load_mnist(validation_size=None, validation_seed=None):
    """
    Get the MNIST training data (downloading it if it is not already accessible),
    and return it as NumPy arrays

    Args:
        validation_size: If not None or zero, set this much data aside for a validation set.
            Floats in (0,1) are interpreted as proportions (e.g., 0.1 uses 10% of the data for validation)
            Integers are interpreted as an absolute number of samples (e.g., 5000 puts exactly 5000 samples in the set)
        validation_seed: Seed for shuffling/selecting validation set

    :return: (train_images, train_labels, test_images, test_labels) if validation_proportion=None or zero, otherwise
        returns (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """

    if os.path.exists("data/"):
        print("data folder already exists")
    else:
        print("Creating data folder")
        os.makedirs("data/")

    if not os.path.exists("data/train-images-idx3-ubyte.gz"):
        print("No MNIST training images found--downloading")
        _download_file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
    train_images = _load_image('data/train-images-idx3-ubyte.gz')

    if not os.path.exists("data/train-labels-idx1-ubyte.gz"):
        print("No MNIST training labels found--downloading")
        _download_file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
    train_labels = _load_label('data/train-labels-idx1-ubyte.gz')

    if not os.path.exists("data/t10k-images-idx3-ubyte.gz"):
        print("No MNIST test (10k) images found--downloading")
        _download_file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
    test_images = _load_image("data/t10k-images-idx3-ubyte.gz")

    if not os.path.exists("data/t10k-labels-idx1-ubyte.gz"):
        print("No MNIST test (10k) labels found--downloading")
        _download_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
    test_labels = _load_label('data/t10k-labels-idx1-ubyte.gz')

    if validation_size is None or validation_size == 0:
        return train_images, train_labels, test_images, test_labels
    else:
        (train_images, val_images, train_labels, val_labels) = train_test_split(train_images, train_labels,
                                                                                random_state=validation_seed,
                                                                                test_size=validation_size,
                                                                                stratify=train_labels)
        return train_images, train_labels, test_images, test_labels, val_images, val_labels


def _download_file(file_path, folder='data'):
    print("Downloading {}...".format(file_path))

    test_file = urllib.URLopener()
    file_name = file_path.split('/')[-1]
    test_file.retrieve(file_path, '{}/{}'.format(folder, file_name))


def _load_image(filename):
    # Read the inputs in Yann LeCun's binary format.
    f = gzip.open(filename, 'rb')
    data = np.frombuffer(f.read(), np.uint8, offset=16)
    f.close()

    data = data.reshape(-1, 784)
    return data / np.float32(256)


def _load_label(filename):
    """Read the labels in Yann LeCun's binary format."""
    f = gzip.open(filename, 'rb')
    data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data


def main():
    """Main does a simple checks"""
    EXAMPLE_IMAGE = 0
    MNIST_LOADERS = [[load_mnist, 'MNIST'], [load_fashion_mnist, "Fashion MNIST"]]

    for (fcn, label) in MNIST_LOADERS:

        (train_images, train_labels, test_images, test_labels) = fcn()
        print('\n== {} =='.format(label))
        print('* Loaded {} training examples.\n* Loaded {} test items.'.format(train_images.shape[0],
                                                                               test_images.shape[0]))
        print('Each image is {} pixels.\n'.format('x'.join([str(x) for x in train_images.shape[1:]])))

        # ASCII graphics for the win
        for x in range(28):
            s = [' '] * 28
            for y in range(28):
                if train_images[EXAMPLE_IMAGE, x*28 + y] > 0.5:
                    s[y] = '#'
            print ''.join(s)
        print("Label: {}\n".format(train_labels[EXAMPLE_IMAGE]))

        # MNIST.train is 60000 x 784
        assert(len(train_images.shape) == 2)
        assert(train_images.shape[0] == 60000 and train_images.shape[1] == 784)
        assert(len(train_labels.shape) == 1 and train_labels.shape[0] == train_images.shape[0])

        # MNIST.test  is 10000 x 784
        assert(len(test_images.shape) == 2)
        assert(test_images.shape[0] == 10000 and test_images.shape[1] == 784)
        assert(len(test_labels.shape) == 1 and test_labels.shape[0] == test_images.shape[0])

        (train_images, train_labels, test_images, test_labels, val_images, val_labels) = load_mnist(validation_size=0.10)
        print("\n== {} with 10% validation set ==".format(label))
        print('* Loaded {} training examples\n* Loaded {} validation items\n* Loaded {} test items'.format(train_images.shape[0],
                                                                                                           val_images.shape[0],
                                                                                                           test_images.shape[0]))
        print('Each image is {} pixels.\n'.format('x'.join([str(x) for x in val_images.shape[1:]])))
        assert (len(val_images.shape) == 2)
        assert (val_images.shape[0] == 6000 and val_images.shape[1] == 784)
        assert (len(val_labels.shape) == 1 and val_labels.shape[0] == val_images.shape[0])


if __name__ == "__main__":
    main()
