from sklearn.utils import shuffle

from net_tf import Network
from vulcanai.utils import get_one_hot
from vulcanai import mnist_loader

import os

import tensorflow as tf

import tf_utils

import numpy as np

(train_images, train_labels, test_images, test_labels) = mnist_loader.load_fashion_mnist()

train_images, train_labels = shuffle(train_images, train_labels, random_state=0)

# f_net = Network.load_model('models/20170828235548_fashion.network')
# m_net = Network.load_model('models/20170828235251_mnist.network')
# f_max_net = Network.load_model('models/20170902174725_1_dense_max.network')

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


train_labels = get_one_hot(train_labels)
test_labels = get_one_hot(test_labels)
print len(train_images[0])
train_images = tf.reshape(train_images, (train_images.shape[0], 28, 28))
test_images = tf.reshape(test_images, (test_images.shape[0], 28, 28))

print train_images.shape

with tf.Graph().as_default() as tf_graph:
    sess = tf.InteractiveSession()
    input_var = tf.placeholder(tf.float32, shape=([None, 28,28,1]),
                               name='input')
    y = tf.placeholder(tf.float32, shape=([None, 10]),
                       name='truth')

    network_conv_config = {
        'mode': 'conv',
        'filters': [16, 32],
        'filter_size': [[5, 5], [5, 5]],
        'stride': [[1, 1], [1, 1]],
        'pool': {
            'mode': 'MAX',  # 'MAX' or 'AVG'
            'stride': [[2, 2], [2, 2]]
        }
    }

    network_dense_config = {
        'mode': 'dense',
        'units': [512],
        'dropouts': [0.3],
    }
    with tf.variable_scope("conv_net"):
        conv_net = Network(
            name='conv_test',
            dimensions=[None, 1] + list(train_images.shape[1:]),
            input_var=input_var,
            y=y,
            config=network_conv_config,
            input_network=None,
            num_classes=None
            )

    with tf.variable_scope("dense_net"):
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

train_images = np.expand_dims(train_images, axis=1)
test_images = np.expand_dims(test_images, axis=1)
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
