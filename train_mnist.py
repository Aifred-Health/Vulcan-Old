import numpy as np

from src.net import Network

import theano.tensor as T

from src.utils import get_one_hot

from src import mnist_loader

from src.model_tests import run_test

(train_images, train_labels, test_images, test_labels) = mnist_loader.load_fashion_mnist()

train_labels = get_one_hot(train_labels)

input_var = T.fmatrix('input')
y = T.fmatrix('truth')

network_dense_config = {
    'mode': 'dense',
    'units': [1024, 1024, 784],
    'dropouts': [0.2, 0.2, 0.2],
}

dense_net = Network(
    name='3_dense_test',
    dimensions=[None] + list(train_images.shape[1:]),
    input_var=input_var,
    y=y,
    config=network_dense_config,
    input_network=None,
    num_classes=10,
    activation='rectify',
    pred_activation='softmax',
    optimizer='adam')

# # Use to load model from disk
# # dense_net = Network.load_model('models/20170704194033_3_dense_test.network')
dense_net.train(
    epochs=2,
    train_x=train_images[:50000],
    train_y=train_labels[:50000],
    val_x=train_images[50000:60000],
    val_y=train_labels[50000:60000],
    batch_ratio=0.05,
    plot=True
)

dense_net.save_record()

run_test(dense_net, test_x=train_images[50000:60000], test_y=train_labels[50000:60000])
dense_net.save_model()
