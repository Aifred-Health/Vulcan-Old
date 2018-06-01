
from net_tf import Network
from vulcanai.utils import get_one_hot
from vulcanai import mnist_loader

import tensorflow as tf
import tf_utils

(train_images, train_labels, test_images, test_labels) = mnist_loader.load_fashion_mnist()

train_labels = get_one_hot(train_labels)


network_dense_config = {
    'mode': 'dense',
    'units': [512, 100],
    'dropouts': [0.2, .05],
}

with tf.Graph().as_default():
    sess = tf.InteractiveSession()
    input_var, y = tf_utils.init_placeholders(len(train_images[0]), len(train_labels[0]))

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
