import numpy as np

from src.net import Network

import theano.tensor as T

from src.utils import get_one_hot

from src import mnist_loader

from src.model_tests import run_test

from src.utils import display_tsne

(train_images, train_labels, test_images, test_labels) = mnist_loader.load_fashion_mnist()

from sklearn.utils import shuffle

train_images, train_labels = shuffle(train_images, train_labels, random_state=0)

# f_net = Network.load_model('models/20170828235548_fashion.network')
# m_net = Network.load_model('models/20170828235251_mnist.network')

label_map = {
    '0': 'T-shirt/top',
    '1': 'Trouser',
    '2': 'Pullover',
    '3': 'Dress',
    '4': 'Coat',
    '5': 'Sandal',
    '6': 'Shirt',
    '7': 'Sneaker',
    '8': 'Bag',
    '9': 'Ankle boot'
}

display_tsne(train_images[:3000], train_labels[:3000], label_map)

train_labels = get_one_hot(train_labels)
test_labels = get_one_hot(test_labels)

train_images = np.reshape(train_images, (train_images.shape[0], 28, 28))
test_images = np.reshape(test_images, (test_images.shape[0], 28, 28))

train_images = np.expand_dims(train_images, axis=1)
test_images = np.expand_dims(test_images, axis=1)

input_var = T.tensor4('input')
y = T.fmatrix('truth')

network_conv_config = {
    'mode': 'conv',
    'filters': [16, 32],
    'filter_size': [[5, 5], [5, 5]],
    'stride': [[1, 1], [1, 1]],
    'pool': {
        'mode': 'average_exc_pad',
        'stride': [[2, 2], [2, 2]]
    }
}

network_dense_config = {
    'mode': 'dense',
    'units': [512],
    'dropouts': [0.3],
}

conv_net = Network(
    name='conv_test',
    dimensions=[None, 1] + list(train_images.shape[1:]),
    input_var=input_var,
    y=y,
    config=network_conv_config,
    input_network=None,
    num_classes=None)

dense_net = Network(
    name='1_dense',
    dimensions=(None, int(train_images.shape[1])),
    input_var=input_var,
    y=y,
    config=network_dense_config,
    input_network={'network': conv_net, 'layer': 4, 'get_params': True},
    num_classes=10,
    activation='rectify',
    pred_activation='softmax'
)
# # Use to load model from disk
# # dense_net = Network.load_model('models/20170704194033_3_dense_test.network')
dense_net.train(
    epochs=200,
    train_x=train_images[:50000],
    train_y=train_labels[:50000],
    val_x=train_images[50000:60000],
    val_y=train_labels[50000:60000],
    batch_ratio=0.05,
    plot=False
)

dense_net.save_record()

run_test(dense_net, test_x=test_images, test_y=test_labels)
dense_net.save_model()
