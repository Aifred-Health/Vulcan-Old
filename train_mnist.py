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


if os.path.exists("data/"):
    print "data folder already exists"
else:
    print "Creating data folder"
    os.makedirs("data/")

if os.path.exists("data/train-images-idx3-ubyte.npy"):
    print "MNIST train images already exist."
    train_images = np.load("data/train-images-idx3-ubyte.npy")
else:
    print "Downloading MNIST train images..."
    testfile = urllib.URLopener()
    testfile.retrieve("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz")
    print("MNIST train images download complete.")
    f = gzip.open('train-images-idx3-ubyte.gz', 'r')
    outf = open("train-images-idx3-ubyte", 'wb')
    outf.write(f.read())
    f.close()
    outf.close()
    os.remove('train-images-idx3-ubyte.gz')
    shutil.move('train-images-idx3-ubyte', 'data/train-images-idx3-ubyte')
    arr1 = np.fromfile('data/train-images-idx3-ubyte', dtype=np.float32)
    np.save('data/train-images-idx3-ubyte.npy', arr1)
    train_images = np.load('data/train-images-idx3-ubyte.npy')
    os.remove('data/train-images-idx3-ubyte')

if os.path.exists("data/train-labels-idx1-ubyte.npy"):
    print "MNIST train labels already exist."
    train_labels = np.load("data/train-labels-idx1-ubyte.npy")
else:
    print "Downloading MNIST train labels..."
    testfile = urllib.URLopener()
    testfile.retrieve("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz")
    print("MNIST train labels download complete.")
    g = gzip.open('train-labels-idx1-ubyte.gz', 'r')
    outg = open("train-labels-idx1-ubyte", 'wb')
    outg.write(g.read())
    g.close()
    outg.close()
    os.remove('train-labels-idx1-ubyte.gz')
    shutil.move('train-labels-idx1-ubyte', 'data/train-labels-idx1-ubyte')
    arr2 = np.fromfile('data/train-labels-idx1-ubyte', dtype=np.float32)
    np.save('data/train-labels-idx1-ubyte.npy', arr2)
    train_labels = np.load('data/train-images-idx3-ubyte.npy')
    os.remove('data/train-labels-idx1-ubyte')

if os.path.exists("data/t10k-images-idx3-ubyte.npy"):
    print "MNIST t10k images already exist."
    t10k_images = np.load("data/t10k-images-idx3-ubyte.npy")
else:
    print "Downloading MNIST t10k images..."
    testfile = urllib.URLopener()
    testfile.retrieve("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz")
    print("MNIST t10k images download complete.")
    h = gzip.open('t10k-images-idx3-ubyte.gz', 'r')
    outh = open("t10k-images-idx3-ubyte", 'wb')
    outh.write(h.read())
    h.close()
    outh.close()
    os.remove('t10k-images-idx3-ubyte.gz')
    shutil.move('t10k-images-idx3-ubyte', 'data/t10k-images-idx3-ubyte')
    arr3 = np.fromfile('data/t10k-images-idx3-ubyte', dtype=np.float32)
    np.save('data/t10k-images-idx3-ubyte.npy', arr3)
    t10k_images = np.load('data/t10k-images-idx3-ubyte.npy')
    os.remove('data/t10k-images-idx3-ubyte')

if os.path.exists("data/t10k-labels-idx1-ubyte.npy"):
    print "MNIST t10k labels exist"
    t10k_labels = np.load("data/t10k-labels-idx1-ubyte.npy")
else:
    print "Downloading MNIST t10k labels..."
    testfile = urllib.URLopener()
    testfile.retrieve("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    print("MNIST t10k labels download complete.")
    k = gzip.open('t10k-labels-idx1-ubyte.gz', 'r')
    outk = open("t10k-labels-idx1-ubyte", 'wb')
    outk.write(k.read())
    k.close()
    outk.close()
    os.remove('t10k-labels-idx1-ubyte.gz')
    shutil.move('t10k-labels-idx1-ubyte', 'data/t10k-labels-idx1-ubyte')
    arr4 = np.fromfile('data/t10k-labels-idx1-ubyte', dtype=np.float32)
    np.save('data/t10k-labels-idx1-ubyte.npy', arr4)
    t10k_labels = np.load('data/t10k-labels-idx1-ubyte.npy')
    os.remove('data/t10k-labels-idx1-ubyte')


print train_images.shape
print train_labels.shape
print t10k_images.shape
print t10k_labels.shape




























