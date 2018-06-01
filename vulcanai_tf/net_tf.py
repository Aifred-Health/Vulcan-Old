import numpy as np

import tensorflow as tf

from vulcanai.ops import activations, optimizers

import cPickle as pickle


class Network(object):
    """Class to generate networks and train them."""

    def __init__(self, name, dimensions, input_var, y, config,
                 input_network=None, num_classes=None, activation='rectify',
                 pred_activation='softmax', optimizer='adam', stopping_rule='best_validation_error',
                 learning_rate=0.001):

        """
        Initialize network specified.

        Args:
            name: string of network name
            dimensions: the size of the input data matrix
            input_var: tensor representing input matrix
            y: tensor representing truth matrix
            config: Network configuration (as dict)
            input_network: None or a dictionary containing keys (network, layer).
                network: a Network object
                layer: an integer corresponding to the layer you want output
            num_classes: None or int. how many classes to predict
            activation:  activation function for hidden layers
            pred_activation: the classifying layer activation
            optimizer: which optimizer to use as the learning function
            learning_rate: the initial learning rate
        """
        self.name = name
        self.input_dim = dimensions
        self.input_var = input_var
        self.y = y
        self.config = config

        self.num_classes = num_classes
        if not optimizers.get(optimizer, False):
            raise ValueError(
                'Invalid optimizer option: {}. '
                'Please choose from:'
                '{}'.format(optimizer, optimizers.keys()))
        if not activations.get(activation, False) or \
           not activations.get(pred_activation, False):
            raise ValueError(
                'Invalid activation option: {} and {}. '
                'Please choose from:'
                '{}'.format(activation, pred_activation, activations.keys()))
        self.activation = activation
        self.pred_activation = pred_activation
        self.optimizer = optimizer

        self.learning_rate = learning_rate
        self.init_learning_rate = learning_rate
        self.stopping_rule = stopping_rule
        
        self.input_network = input_network
        self.layers = []
        self.cost = None
        self.val_cost = None
        self.input_params = None
        if self.input_network is not None:
            if self.input_network.get('network', False) is not False and \
               self.input_network.get('layer', False) is not False and \
               self.input_network.get('get_params', None) is not None:
                self.input_var = None   # NOTE: Write a Tensorflow replacement code
                self.input_dim = None   # NOTE: Write a Tensorflow replacement code
                if self.input_network.get('get_params', False):
                    self.input_params = self.input_network['network'].params

            else:
                raise ValueError(
                    'input_network for {} requires {{ network: type Network,'
                    ' layer: type int, get_params: type bool}}. '
                    'Only given keys: {}'.format(
                        self.name, self.input_network.keys()
                    )
                )

        self.network = self.create_network(
            config=self.config,
            nonlinearity=activations[self.activation])

        #if self.y is not None:
        #    self.trainer =
        #    self.validator


    def create_network(self, config, nonlinearity):

        import jsonschema
        import schemas
        mode = config.get('mode')
        if mode == 'dense':
            jsonschema.validate(config, schemas.dense_network)

            network = self.build_dense_network_tf(
                units=config.get('units'),
                dropouts=config.get('dropouts'),
                nonlinearity=nonlinearity
            )
        elif mode == 'conv':
            jsonschema.validate(config, schemas.conv_network)

            network = self.create_conv_network(
                filters=config.get('filters'),
                filter_size=config.get('filter_size'),
                stride=config.get('stride'),
                pool_mode=config['pool'].get('mode'),
                pool_stride=config['pool'].get('stride'),
                nonlinearity=nonlinearity
            )
        else:
            raise ValueError('Mode {} not supported.'.format(mode))

        if self.num_classes is not None and self.num_classes != 0:
            network = self.create_classification_layer(
                network,
                num_classes=self.num_classes,
                nonlinearity=activations[self.pred_activation]
            )

        return network

    def create_conv_network(self, filters, filter_size, stride,
                            pool_mode, pool_stride, nonlinearity):
        """
        Create a convolutional network (1D, 2D, or 3D).

        Args:
            filters: list of int. number of kernels per layer
            filter_size: list of int list. size of kernels per layer
            stride: list of int list. stride of kernels
            pool_mode: string. pooling operation
            pool_stride: list of int list. down_scaling factor
            nonlinearity: string. nonlinearity to use for each layer

        Returns a conv network
        """
        conv_dim = len(filter_size[0])
        tf_pools = ['MAX', 'AVG']
        if not all(len(f) == conv_dim for f in filter_size):
            raise ValueError('Each tuple in filter_size {} must have a '
                             'length of {}'.format(filter_size, conv_dim))
        if not all(len(s) == conv_dim for s in stride):
            raise ValueError('Each tuple in stride {} must have a '
                             'length of {}'.format(stride, conv_dim))
        if not all(len(p) == conv_dim for p in pool_stride):
            raise ValueError('Each tuple in pool_stride {} must have a '
                             'length of {}'.format(pool_stride, conv_dim))
        if pool_mode not in tf_pools:
            raise ValueError('{} pooling does not exist. '
                             'Please use one of: {}'.format(pool_mode, tf_pools))

        print("Creating {} Network...".format(self.name))

        if self.input_network is None:
            with tf.variable_scope('Input_layer'):
                print('\tInput Layer:')
                network = tf.keras.layers.InputLayer(
                                    input_shape=self.input_dim,
                                    input_tensor=self.input_var,
                                    name="{}_input"
                                    .format(self.name)).output
                print('\t\t{} {}'.format(network.shape, network.name))
                self.layers.append(network)
        else:
            with tf.variable_scope('prev_layer'):
                network = self.input_network['network']. \
                    layers[self.input_network['layer']]

                print('Appending layer {} from {} to {}'.format(
                    self.input_network['layer'],
                    self.input_network['network'].name,
                    self.name))

        if conv_dim == 1:
            conv_layer = tf.layers.conv1d
            if pool_mode == 'AVG':
                pool = tf.layers.average_pooling1d
            else:
                pool = tf.layers.max_pooling1d
        elif conv_dim == 2:
            conv_layer = tf.layers.conv2d
            if pool_mode == 'AVG':
                pool = tf.layers.average_pooling2d
            else:
                pool = tf.layers.max_pooling2d
        elif conv_dim == 3:
            conv_layer = tf.layers.conv2d
            if pool_mode == 'AVG':
                pool = tf.layers.average_pooling1d
            else:
                pool = tf.layers.max_pooling1d
        else:
            pool = None   # Linter is stupid
            conv_layer = None
            ValueError("Convolution is only supported for one of the first three dimensions")

        print('\tHidden Layers:')
        for i, (f, f_size, s, p_s) in enumerate(zip(filters,
                                                    filter_size,
                                                    stride,
                                                    pool_stride)):
            layer_name = "conv{}D_layer{}".format(
                conv_dim, i)
            with tf.variable_scope(layer_name):
                network = conv_layer(
                    inputs=network,
                    filters=f,
                    kernel_size=f_size,
                    strides=s,
                    padding='same',
                    activation=nonlinearity,
                    name="{}_conv{}D_{}".format(
                        self.name, conv_dim, i)
                )
                self.layers.append(network)
                print('\t\t{} {}'.format(network.shape, network.name))
                network = pool(
                    inputs=network,
                    pool_size=p_s,
                    strides=p_s,
                    padding='same',
                    name="{}_{}pool".format(
                        self.name, pool_mode)
                )
            self.layers.append(network)
            print('\t\t{} {}'.format(network.shape, network.name))
        return network

    def create_dense_network_tf(self, units, dropouts, nonlinearity):
        """
        Generate a fully connected network of dense layers.

        Args:
            units: The list of number of nodes to have at each layer
            dropouts: The list of dropout probabilities for each layer
            nonlinearity: Nonlinearity from Tensorflow.nn

        Returns: the output of the network (linked up to all the layers)
        """
        if len(units) != len(dropouts):
            raise ValueError(
                "Cannot build network: units and dropouts don't correspond"
            )

        print("Creating {} Network...".format(self.name))
        if self.input_network is None:
            with tf.variable_scope('Input_layer'):
                print('\tInput Layer:')
                network = tf.keras.layers.InputLayer(
                                    input_shape=self.input_dim,
                                    input_tensor=self.input_var,
                                    name="{}_input"
                                    .format(self.name)).output
                print('\t\t{} {}'.format(network.shape, network.name))
                self.layers.append(network)
        else:
            with tf.variable_scope('prev_layer'):
                network = self.input_network['network']. \
                    layers[self.input_network['layer']]

                print('Appending layer {} from {} to {}'.format(
                    self.input_network['layer'],
                    self.input_network['network'].name,
                    self.name))

        if nonlinearity.__name__ == 'selu':
            with tf.variable_scope('batch_norm'):
                network = tf.layers.batch_normalization(
                            network,
                            training=(mode == tf.estimator.ModeKeys.TRAIN),
                            name="{}_batchnorm".format(self.name))

        print('\tHidden Layers:')
        for i, (num_units, prob_dropout) in enumerate(zip(units, dropouts)):
            layer_name = 'dense_layer%s' % i
            with tf.variable_scope(layer_name):
                if nonlinearity.__name__ == 'selu':
                    with tf.variable_scope('weights'):
                        w = tf.random_normal(
                                shape=[network.shape[1], num_units],
                                stddev=np.sqrt(1.0 / num_units),
                                name='w_selu')
                        tf.summary.histogram(layer_name + '/weights', w)
                    with tf.variable_scope('biases'):
                        b = tf.random_normal(
                                shape=[1, num_units],
                                stddev=0.0,
                                name='b_selu')
                        tf.summary.histogram(layer_name + '/biases', b)

                else:
                    with tf.variable_scope('weights'):
                        w = tf.get_variable(
                                shape=[network.shape[1], num_units],
                                initializer=tf.glorot_uniform_initializer(),
                                name='w')
                        tf.summary.histogram(layer_name + '/weights', w)
                    with tf.variable_scope('biases'):
                        b = tf.get_variable(
                                shape=[1, num_units],
                                initializer=tf.glorot_uniform_initializer(),
                                name='b')
                        tf.summary.histogram(layer_name + '/biases', b)

                new_layer = nonlinearity(tf.add(tf.matmul(network, w), b))

                if nonlinearity.__name__ == 'selu':
                    network = tf.contrib.nn.alpha_dropout(new_layer,
                                                    prob_dropout)
                else:
                    network = tf.nn.dropout(new_layer,
                                                    prob_dropout)

                tf.summary.histogram(layer_name + '/outputs', network)

            self.layers.append(network)
            print('\t\t{} {}'.format(network.shape,network.name))
        return network

    def build_dense_network_tf(self, units, dropouts, nonlinearity):
        """
        Generate a fully connected network of dense layers.

        Args:
            units: The list of number of nodes to have at each layer
            dropouts: The list of dropout probabilities for each layer
            nonlinearity: Nonlinearity from Tensorflow.nn

        Returns: the output of the network (linked up to all the layers)
        """
        if len(units) != len(dropouts):
            raise ValueError(
                "Cannot build network: units and dropouts don't correspond"
            )

        print("Creating {} Network...".format(self.name))
        if self.input_network is None:
            with tf.variable_scope('Input_layer'):
                print('\tInput Layer:')
                network = tf.keras.layers.InputLayer(
                                    input_shape=self.input_dim,
                                    input_tensor=self.input_var,
                                    name="{}_input"
                                    .format(self.name)).output
                print('\t\t{} {}'.format(network.shape, network.name))
                self.layers.append(network)
        else:
            with tf.variable_scope('prev_layer'):
                network = self.input_network['network']. \
                    layers[self.input_network['layer']]

                print('Appending layer {} from {} to {}'.format(
                    self.input_network['layer'],
                    self.input_network['network'].name,
                    self.name))

        if nonlinearity.__name__ == 'selu':
            network = tf.layers.batch_normalization(
                        network,
                        training=(mode == tf.estimator.ModeKeys.TRAIN),
                        name="{}_batchnorm".format(self.name))

        print('\tHidden Layers:')
        for i, (num_units, prob_dropout) in enumerate(zip(units, dropouts)):
            layer_name = 'dense_layer1%s' % i
            with tf.variable_scope(layer_name):
                if nonlinearity.__name__ == 'selu':
                    new_layer = tf.layers.dense(
                                        inputs=network,
                                        units=num_units,
                                        activation=nonlinearity,
                                        kernel_initializer=tf.initializers
                                        .random_normal(stddev=np.sqrt(
                                            1.0 / num_units)),
                                        bias_regularizer=tf.initializers
                                        .random_normal(stddev=0.0),
                                        name="dense_selu")

                    network = tf.contrib.nn.alpha_dropout(new_layer,
                                                        prob_dropout)
                    tf.summary.histogram(layer_name + '/selu', network)
                else:
                    new_layer = tf.layers.dense(
                                        inputs=network,
                                        units=num_units,
                                        activation=nonlinearity,
                                        name="dense_layer")  # By default TF assumes Glorot uniform initializer for weights and zero initializer for bias

                    network = tf.nn.dropout(new_layer,
                                                    prob_dropout)
                    tf.summary.histogram(layer_name+'/'+self.activation, network)
#                network = tf.Print(network, [tf.argmax(network, 1)],
#                   'argmax(out) = ', summarize=10, first_n=7)
            self.layers.append(network)
            print('\t\t{} {}'.format(network.shape, network.name))

        return network

    def create_classification_layer(self, network, num_classes,
                                    nonlinearity):
        """
        Create a classification layer. Normally used as the last layer.

        Args:
            network: network you want to append a classification to
            num_classes: how many classes you want to predict
            nonlinearity: nonlinearity to use as a string (see DenseLayer)

        Returns: the classification layer appended to all previous layers
        """
        print('\tOutput Layer:')
        network = tf.layers.dense(
                            inputs=network,
                            units=num_classes,
                            activation=tf.nn.softmax,
                            name="last_layer")

        print('\t\t{}'.format(network.shape))
        self.layers.append(network)
        return network

    def cross_entropy_loss(self, prediction, y):
        """Generate a cross entropy loss function."""
        print("Using categorical cross entropy loss")
        return  # NOTE: Write a Tensorflow replacement code

    def mse_loss(self, prediction, y):
        """Generate mean squared error loss function."""
        print("Using Mean Squared error loss")
        return  # NOTE: Write a Tensorflow replacement code

    @classmethod
    def load_model(cls, load_path):
        """
        Will load the model parameters from npz file.

        Args:
            load_path: the exact location where the model has been saved.
        """
        print('Loading model from: {}'.format(load_path))
        with open(load_path, 'rb') as f:
            instance = pickle.load(f)
        return instance



if __name__ == "__main__":
    pass
