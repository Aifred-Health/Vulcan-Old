import theano.tensor as T

from vulcanai.net import Network
from vulcanai.utils import get_one_hot
from vulcanai.model_tests import run_test
from vulcanai import mnist_loader

(train_images, train_labels, test_images, test_labels) = mnist_loader.load_mnist()

train_labels = get_one_hot(train_labels)

input_var = T.fmatrix('input')
y = T.fmatrix('truth')

network_auto_config = {
    'mode': 'dense',
    'units': [784, 392, 784],
    'dropouts': [0.2, 0.2, 0.2],
}

autoencoder = Network(
    name='autoencoder_mnist',
    dimensions=(None, int(train_images.shape[1])),
    input_var=input_var,
    y=y,
    config=network_auto_config,
    input_network=None,
    num_classes=None
)
# Use to load model from disk
# autoencoder = Network.load_model('models/20170701174206_autoencoder_mnist.network')

autoencoder.train(
    epochs=2,
    train_x=train_images[:50000],
    train_y=train_images[:50000],
    val_x=train_images[50000:60000],
    val_y=train_images[50000:60000],
    batch_ratio=0.5,
    plot=True
)

network_dense_config = {
    'mode': 'dense',
    'units': [1024, 1024, 784],
    'dropouts': [0.2, 0.2, 0.2],
}

dense_net = Network(
    name='3_dense',
    dimensions=(None, int(train_images.shape[1])),
    input_var=input_var,
    y=y,
    config=network_dense_config,
    input_network={'network': autoencoder, 'layer': 4, 'get_params': False},
    num_classes=10
)

dense_net.train(
    epochs=3,
    train_x=train_images[:50000],
    train_y=train_labels[:50000],
    val_x=train_images[50000:60000],
    val_y=train_labels[50000:60000],
    batch_ratio=0.5,
    plot=True
)

run_test(dense_net, test_x=train_images[50000:60000], test_y=train_labels[50000:60000])
# saves all embedded networks too if they are not allready save
dense_net.save_model()
