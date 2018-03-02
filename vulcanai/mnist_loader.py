import os
import urllib
import gzip
import numpy as np


def load_fashion_mnist():
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
    t10k_images = _load_image("data/fashion/t10k-images-idx3-ubyte.gz")

    if not os.path.exists("data/fashion/t10k-labels-idx1-ubyte.gz"):
        print("No fashion MNIST test (10k) labels found--downloading")
        _download_file("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", 'data/fashion')
    print ('Loading testing labels...')
    t10k_labels = _load_label('data/fashion/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, t10k_images, t10k_labels


def load_mnist():
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
    t10k_images = _load_image("data/t10k-images-idx3-ubyte.gz")

    if not os.path.exists("data/t10k-labels-idx1-ubyte.gz"):
        print("No MNIST test (10k) labels found--downloading")
        _download_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
    print ('Loading testing labels...')
    t10k_labels = _load_label('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, t10k_images, t10k_labels


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
    """Totally useless main."""
    (train_iamges, train_labels, t10k_images, t10k_labels) = load_mnist()

if __name__ == "__main__":
    main()
