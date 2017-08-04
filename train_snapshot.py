import numpy as np

import theano.tensor as T

from src.net import Network

from src.snapshot_ensemble import Snapshot

from src.utils import get_one_hot

from src import mnist_loader

from src.model_tests import run_test

(train_images, train_labels, test_images, test_labels) = mnist_loader.load_mnist()


train_labels = get_one_hot(train_labels)

input_var = T.fmatrix('input')
y = T.fmatrix('truth')

dense_net = Network(
    name='3_dense_test',
    dimensions=(None, int(train_images.shape[1])),
    input_var=input_var,
    y=y,
    units=[784, 784],
    dropouts=[0.2, 0.2],
    input_network=None,
    num_classes=10,
    activation='rectify',
    learning_rate=0.01
)

ensemble_dense = Snapshot(
    name='snap_test',
    template_network=dense_net,
    n_snapshots=2
)

ensemble_dense.train(
    epochs=2,
    train_x=train_images[:50000],
    train_y=train_labels[:50000],
    val_x=train_images[50000:60000],
    val_y=train_labels[50000:60000],
    batch_ratio=0.05,
    plot=True
)


# ensemble_dense = Snapshot.load_ensemble('models/20170713183810_snap1')
run_test(ensemble_dense, test_x=train_images[50000:60000], test_y=train_labels[50000:60000])
ensemble_dense.save_model()
