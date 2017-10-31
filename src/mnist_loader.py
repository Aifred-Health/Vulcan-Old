import os
import urllib
import gzip
import numpy as np


def load_fashion_mnist(validation_proportion=None, validation_seed=None):
    """
    Get the fashion MNIST training data (downloading it if it is not already accessible),
    and return it as NumPy arrays.

    Extracted from https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md

    :return: (train_images, train_labels, test_images, test_labels)
    """
    if os.path.exists("data/fashion"):
        print("data folder already exists")
    else:
        print("Creating data/fashion folder")
        os.makedirs("data/fashion")

    if not os.path.exists("data/fashion/train-images-idx3-ubyte.gz"):
        print("No fashion MNIST training images found--downloading")
        _download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", 'data/fashion')
    print ('Loading training images...')
    train_images = _load_image('data/fashion/train-images-idx3-ubyte.gz')

    if not os.path.exists("data/fashion/train-labels-idx1-ubyte.gz"):
        print("No fashion MNIST training labels found--downloading")
        _download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",'data/fashion')
    print ('Loading training labels...')
    train_labels = _load_label('data/fashion/train-labels-idx1-ubyte.gz')

    if not os.path.exists("data/fashion/t10k-images-idx3-ubyte.gz"):
        print("No fashion MNIST test (10k) images found--downloading")
        _download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", 'data/fashion')
    print ('Loading testing images...')
    test_images = _load_image("data/fashion/t10k-images-idx3-ubyte.gz")

    if not os.path.exists("data/fashion/t10k-labels-idx1-ubyte.gz"):
        print("No fashion MNIST test (10k) labels found--downloading")
        _download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", 'data/fashion')
    print ('Loading testing labels...')
    test_labels = _load_label('data/fashion/t10k-labels-idx1-ubyte.gz')

    if validation_proportion is None or validation_proportion == 0:
        return train_images, train_labels, test_images, test_labels
    else:
        (train_images, train_labels, val_images, val_labels) = _split_dataset(train_images, train_labels,
                                                                              val_prop=validation_proportion,
                                                                              seed=validation_seed)
        return train_images, train_labels, val_images, val_labels, test_images, test_labels


def load_mnist(validation_proportion=None, validation_seed=None):
    """
    Get the MNIST training data (downloading it if it is not already accessible),
    and return it as NumPy arrays

    :return: (train_images, train_labels, test_images, test_labels)
    """
    if os.path.exists("data/"):
        print("data folder already exists")
    else:
        print("Creating data folder")
        os.makedirs("data/")

    if not os.path.exists("data/train-images-idx3-ubyte.gz"):
        print("No MNIST training images found--downloading")
        _download_file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
    print ('Loading training images...')
    train_images = _load_image('data/train-images-idx3-ubyte.gz')

    if not os.path.exists("data/train-labels-idx1-ubyte.gz"):
        print("No MNIST training labels found--downloading")
        _download_file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
    print ('Loading training labels...')
    train_labels = _load_label('data/train-labels-idx1-ubyte.gz')

    if not os.path.exists("data/t10k-images-idx3-ubyte.gz"):
        print("No MNIST test (10k) images found--downloading")
        _download_file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
    print ('Loading testing images...')
    test_images = _load_image("data/t10k-images-idx3-ubyte.gz")

    if not os.path.exists("data/t10k-labels-idx1-ubyte.gz"):
        print("No MNIST test (10k) labels found--downloading")
        _download_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
    print ('Loading testing labels...')
    test_labels = _load_label('data/t10k-labels-idx1-ubyte.gz')

    if validation_proportion is None or validation_proportion == 0:
        return train_images, train_labels, test_images, test_labels
    else:
        (train_images, train_labels, val_images, val_labels) = _split_dataset(train_images, train_labels,
                                                                              val_prop=validation_proportion,
                                                                              seed=validation_seed)
        return train_images, train_labels, val_images, val_labels, test_images, test_labels


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


def _split_dataset(data, labels, val_prop, seed=None):
    """Split a data set in two (i.e., train and validation)
    Args:
        data:  Data set (examples X dims)
        labels:  Corresponding class labels (examples X 1)
        val_prop: Proportion of examples reserved for the validation set (0-1)
        seed: If not None, this int/array will be used to seed the RNG that selects the samples
    Returns:
        (train_data, train_labels, val_data, val_labels)

    Note: The data in each new set is a copy of the original, not a view

    """
    rng = np.random.RandomState(seed)
    ix = rng.permutation(data.shape[0])

    n_val = np.floor(data.shape[0] * val_prop).astype('int')
    return data[ix[n_val:], :], labels[ix[n_val:]].copy(), data[ix[:n_val], :].copy(), labels[ix[:n_val]].copy(),




def main():
    """Totally useless main."""
    (train_iamges, train_labels, t10k_images, t10k_labels) = load_mnist()
    (train_iamges, train_labels, val_images, val_labels, test_imgages, test_labels) = load_mnist(validation_proportion=0.1)

if __name__ == "__main__":
    main()
