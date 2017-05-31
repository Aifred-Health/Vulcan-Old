import numpy as np

import pandas as pd

import theano.tensor as T

from src.net import Network

import wget

train_image_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
train_label_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
test_image_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
test_label_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

if not os.path.exists('data'):
    print ('Creating data folder')
    os.makedirs('data')
