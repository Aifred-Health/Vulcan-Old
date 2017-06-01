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

def extract_contents(file_path):
    print "Downloading MNIST train images..."
    testfile = urllib.URLopener()
    file_name = file_path.split('/')[-1]
    testfile.retrieve(file_path, file_name)

    print("MNIST %s download complete." % file_name)
    f = gzip.open(file_name, 'rb')
    outf = open(file_name.split('.')[0], 'wb')
    outf.write(f.read())
    f.close()
    outf.close()
    os.remove(file_name)
    shutil.move(file_name.split('.')[0], 'data/%s' % file_name.split('.')[0])
    arr1 = np.fromfile('data/%s' % file_name.split('.')[0], dtype=np.float32)
    np.save('data/%s.npy' % file_name.split('.')[0], arr1)
    os.remove('data/%s' % file_name.split('.')[0])


if os.path.exists("data/"):
    print "data folder already exists"
else:
    print "Creating data folder"
    os.makedirs("data/")

if os.path.exists("data/train-images-idx3-ubyte.npy"):
    print "MNIST train images already exist."
    train_images = np.load("data/train-images-idx3-ubyte.npy")
else:
    extract_contents("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
    train_images = np.load('data/train-images-idx3-ubyte.npy')

if os.path.exists("data/train-labels-idx1-ubyte.npy"):
    print "MNIST train labels already exist."
    train_labels = np.load("data/train-labels-idx1-ubyte.npy")
else:
    extract_contents("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
    train_labels = np.load('data/train-images-idx3-ubyte.npy')

if os.path.exists("data/t10k-images-idx3-ubyte.npy"):
    print "MNIST t10k images already exist."
    t10k_images = np.load("data/t10k-images-idx3-ubyte.npy")
else:
    extract_contents("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
    t10k_images = np.load('data/t10k-images-idx3-ubyte.npy')

if os.path.exists("data/t10k-labels-idx1-ubyte.npy"):
    print "MNIST t10k labels already exist"
    t10k_labels = np.load("data/t10k-labels-idx1-ubyte.npy")
else:
    extract_contents("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
    t10k_labels = np.load('data/t10k-labels-idx1-ubyte.npy')

print train_images.shape
print train_labels.shape
print t10k_images.shape
print t10k_labels.shape




























