import numpy as np

import pandas as pd

import theano.tensor as T

from src.net import Network

import urllib

import gzip

import os

import shutil
train_images = None
train_labels = None
t10k_images = None
train_labels = None

def download_file(file_path):
    print "Downloading %s..." % file_path

    test_file = urllib.URLopener()
    file_name = file_path.split('/')[-1]
    test_file.retrieve(file_path, 'data/%s' % file_name)

def load_image(filename):
    # Read the inputs in Yann LeCun's binary format.
    f = gzip.open(filename, 'rb')
    data = np.frombuffer(f.read(), np.uint8, offset=16)
    f.close()

    data = data.reshape(-1, 784)
    return data / np.float32(256)

def load_label(filename):
    # Read the labels in Yann LeCun's binary format.
    f = gzip.open(filename, 'rb')
    data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

if os.path.exists("data/"):
    print "data folder already exists"
else:
    print "Creating data folder"
    os.makedirs("data/")

if os.path.exists("data/train-images-idx3-ubyte.gz"):
    print "MNIST train images already exist."
    train_images = load_image("data/train-images-idx3-ubyte.gz")
else:
    download_file("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
    train_images = load_image('data/train-images-idx3-ubyte.gz')

if os.path.exists("data/train-labels-idx1-ubyte.gz"):
    print "MNIST train labels already exist."
    train_labels = load_label("data/train-labels-idx1-ubyte.gz")
else:
    download_file("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
    train_labels = load_label('data/train-labels-idx1-ubyte.gz')

if os.path.exists("data/t10k-images-idx3-ubyte.gz"):
    print "MNIST t10k images already exist."
    t10k_images = load_image("data/t10k-images-idx3-ubyte.gz")
else:
    download_file("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
    t10k_images = load_image('data/t10k-images-idx3-ubyte.gz')

if os.path.exists("data/t10k-labels-idx1-ubyte.gz"):
    print "MNIST t10k labels already exist"
    t10k_labels = load_label("data/t10k-labels-idx1-ubyte.gz")
else:
    download_file("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
    t10k_labels = load_label('data/t10k-labels-idx1-ubyte.gz')

print train_images.shape
print train_labels.shape
print t10k_images.shape
print t10k_labels.shape





























