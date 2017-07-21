import numpy as np

import theano.tensor as T

from src.net import Network

from src.snapshot_ensemble import Snapshot

from src.utils import get_one_hot

from src import mnist_loader

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
    n_snapshots=2,
    n_epochs=2,
    batch_ratio=0.005
)
ensemble_dense.train(
    train_x=train_images[:50000],
    train_y=train_labels[:50000],
    val_x=train_images[50000:60000],
    val_y=train_labels[50000:60000],
    plot=True
)
# ensemble_dense.save_ensemble()
# import pudb; pu.db
# ensemble_dense = Snapshot.load_ensemble('models/20170713183810_snap1')
ensemble_dense.conduct_test(test_x=train_images[50000:60000], test_y=train_labels[50000:60000])
# dense_net.train(
#     epochs=6,
#     train_x=train_images[:50000],
#     train_y=train_labels[:50000],
#     val_x=train_images[50000:60000],
#     val_y=train_labels[50000:60000],
#     batch_ratio=0.5,
#     plot=True
# )



# dense_net.save_model()
# dense_net.save_record()

# dense_net.conduct_test(test_x=train_images[50000:60000], test_y=train_labels[50000:60000])
