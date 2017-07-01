import numpy as np

import theano.tensor as T

from src.net import Network

import shutil


from src.utils import get_one_hot

from src import mnist_loader

(train_images, train_labels, test_images, test_labels) = mnist_loader.load_mnist()

train_labels = get_one_hot(train_labels)

input_var = T.fmatrix('input')
y = T.fmatrix('truth')

autoencoder = Network(
    name='autoencoder_mnist',
    dimensions=(None, int(train_images.shape[1])),
    input_var=input_var,
    y=y,
    units=[784, 392, 784],
    dropouts=[0.2, 0.2, 0.2],
    input_network=None,
    num_classes=None
)
# Use to load model from disk
#autoencoder = Network.load_model('models/20170701182012_autoencoder_mnist.network')

autoencoder.train(
    epochs=20,
    train_x=train_images[:50000],
    train_y=train_images[:50000],
    val_x=train_images[50000:60000],
    val_y=train_images[50000:60000],
    batch_ratio=0.5,
    plot=True
)
autoencoder.save_model()

dense_net = Network(
    name='3_dense',
    dimensions=(None, int(train_images.shape[1])),
    input_var=input_var,
    y=y,
    units=[4096, 1024, 784],
    dropouts=[0.2, 0.2, 0.2],
    input_network={'network': autoencoder, 'layer': 4},
    num_classes=10
)

# dense_net = Network.load_model('models/20170701182014_3_dense.network')

dense_net.train(
    epochs=6,
    train_x=train_images[:50000],
    train_y=train_labels[:50000],
    val_x=train_images[50000:60000],
    val_y=train_labels[50000:60000],
    batch_ratio=0.5,
    plot=True
)

dense_net.save_model()

# dense_net.conduct_test(test_x=train_images[50000:60000], test_y=train_labels[50000:60000])
