"""Contains the class for creating networks."""

import time

import os

import sys

import numpy as np

import lasagne

import matplotlib
if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from utils import get_class
from utils import display_record
from utils import get_timestamp

from selu import AlphaDropoutLayer

from model_tests import run_test

from ops import activations, optimizers

import json

import cPickle as pickle

from sklearn.utils import shuffle

sys.setrecursionlimit(5000)


class Network(object):
    """Class to generate networks and train them."""

    def __init__(self, name, dimensions, input_var, y, units, dropouts,
                 input_network=None, num_classes=None, activation='rectify',
                 pred_activation='softmax', optimizer='adam',
                 learning_rate=0.001):
        """
        Initialize network specified.

        Args:
            name: string of network name
            dimensions: the size of the input data matrix
            input_var: theano tensor representing input matrix
            y: theano tensor representing truth matrix
            units: The list of number of nodes to have at each layer
            dropouts: The list of dropout probabilities to have at each layer
            input_network: a dictionary containing keys (network, layer).
                network: a Network object
                layer: an integer corresponding to the layer you want output
            num_classes: None or int. how many classes to predict
            activation: layer activation function
            pred_activation: the classifying layer activation
            optimizer: which optimizer to usem as the learning function
            learning_rate: the initial learning rate
        """
        self.name = name
        self.layers = []
        self.cost = None
        self.input_dimensions = dimensions
        self.units = units
        self.dropouts = dropouts
        self.learning_rate = learning_rate
        self.init_learning_rate = learning_rate
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
        self.input_var = input_var
        self.y = y
        self.input_network = input_network
        if self.input_network is not None:
            if self.input_network.get('network', False) and \
               self.input_network.get('layer', False):

                self.input_var = lasagne.layers.get_all_layers(
                    self.input_network['network']
                )[0].input_var

                self.input_dimensions = lasagne.layers.get_output_shape(
                    self.input_network['network'].layers[
                        self.input_network['layer']
                    ]
                )

            else:
                raise ValueError(
                    'input_network requires {{ network: type Network,'
                    ' layer: type int}}. '
                    'Only given keys: {}'.format(
                        self.input_network.keys()
                    )
                )
        self.network = self.create_dense_network(
            units=units,
            dropouts=dropouts,
            nonlinearity=activations[self.activation]
        )
        self.num_classes = num_classes
        if num_classes is not None and num_classes != 0:
            self.network = self.create_classification_layer(
                self.network,
                num_classes=num_classes,
                nonlinearity=activations[self.pred_activation]
            )
        if self.y is not None:
            self.trainer = self.create_trainer()
            self.validator = self.create_validator()

        self.output = theano.function(
            [i for i in [self.input_var] if i],
            lasagne.layers.get_output(self.network, deterministic=True))
        self.record = None
        try:
            self.timestamp
        except AttributeError:
            self.timestamp = get_timestamp()
        self.minibatch_iteration = 0

    def create_dense_network(self, units, dropouts, nonlinearity):
        """
        Generate a fully connected layer.

        Args:
            dimension: the size of the incoming theano tensor
            input_var: a theano tensor representing your data input
            units: The list of number of nodes to have at each layer
            dropout: The list of dropout probabilities to have at each layer

        Returns: the output of the network (linked up to all the layers)
        """
        if len(units) != len(dropouts):
            print ("Cannot build network: units and dropouts don't correspond")
            return

        print ("Creating {} Network...".format(self.name))
        if self.input_network is None:
            print ('\tInput Layer:')
            network = lasagne.layers.InputLayer(shape=self.input_dimensions,
                                                input_var=self.input_var,
                                                name="{}_input".format(
                                                     self.name))
            print '\t\t', lasagne.layers.get_output_shape(network)
            self.layers.append(network)
        else:
            network = self.input_network['network']. \
                layers[self.input_network['layer']]

            print ('Appending layer {} from {} to {}'.format(
                self.input_network['layer'],
                self.input_network['network'].name,
                self.name))

        print ('\tHidden Layer:')
        for i, (num_units, prob_dropout) in enumerate(zip(units, dropouts)):
            network = lasagne.layers.DenseLayer(
                incoming=network,
                num_units=num_units,
                nonlinearity=nonlinearity,
                name="{}_dense_{}".format(self.name, i)
            )
            network.add_param(
                network.W,
                network.W.get_value().shape,
                **{self.name: True}
            )
            network.add_param(
                network.b,
                network.b.get_value().shape,
                **{self.name: True}
            )
            self.layers.append(network)

            if nonlinearity.__name__ == 'selu':
                network = AlphaDropoutLayer(incoming=network)
            else:
                network = lasagne.layers.DropoutLayer(
                    incoming=network,
                    p=prob_dropout,
                    name="{}_dropout_{}".format(self.name, i)
                )

            self.layers.append(network)
            print '\t\t', lasagne.layers.get_output_shape(network)

        return network

    def create_classification_layer(self, network, num_classes,
                                    nonlinearity):
        """
        Create a classificatino layer. Normally used as the last layer.

        Args:
            network: network you want to append a classification to
            num_classes: how many classes you want to predict

        Returns: the classification layer appended to all previous layers
        """
        print ('\tOutput Layer:')
        network = lasagne.layers.DenseLayer(
            incoming=network,
            num_units=num_classes,
            nonlinearity=nonlinearity,
            name="{}_softmax".format(self.name)
        )
        network.add_param(
            network.W,
            network.W.get_value().shape,
            **{self.name: True}
        )
        network.add_param(
            network.b,
            network.b.get_value().shape,
            **{self.name: True}
        )
        print '\t\t', lasagne.layers.get_output_shape(network)
        self.layers.append(network)
        return network

    def cross_entropy_loss(self, prediction, y):
        """Generate a cross entropy loss function."""
        print ("Using categorical cross entropy loss")
        return lasagne.objectives.categorical_crossentropy(prediction,
                                                           y).mean()

    def mse_loss(self, prediction, y):
        """Generate mean squared error loss function."""
        print ("Using Mean Squared error loss")
        return lasagne.objectives.squared_error(prediction, y).mean()

    def create_trainer(self):
        """
        Generate a theano function to train the network.

        Args:
            network: Lasagne object representing the network
            input_var: theano.tensor object used for data input
            y: theano.tensor object used for truths

        Returns: theano function that takes as input (train_x,train_y)
                 and trains the net
        """
        print ("Creating {} Trainer...".format(self.name))
        # get network output
        out = lasagne.layers.get_output(self.network)
        # get all trainable parameters from network

        self.params = lasagne.layers.get_all_params(
            self.network,
            trainable=True,
            **{self.name: True}
        )
        # calculate a loss function which has to be a scalar
        if self.cost is None:
            if self.num_classes is None or self.num_classes == 0:
                self.cost = self.mse_loss(out, self.y)
            else:
                self.cost = self.cross_entropy_loss(out, self.y)

        # calculate updates using ADAM optimization gradient descent
        learning_rate_var = T.scalar(name='learning_rate')
        if self.optimizer == 'adam':
            updates = optimizers[self.optimizer](
                loss_or_grads=self.cost,
                params=self.params,
                learning_rate=learning_rate_var,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
        elif self.optimizer == 'sgd':
            updates = optimizers[self.optimizer](
                loss_or_grads=self.cost,
                params=self.params,
                learning_rate=learning_rate_var
            )

        # omitted (, allow_input_downcast=True)
        return theano.function(
            [i for i in [self.input_var, self.y, learning_rate_var] if i],
            updates=updates
        )

    def create_validator(self):
        """
        Generate theano function to check error and accuracy of the network.

        Args:
            network: Lasagne object representing the network
            input_var: theano.tensor object used for data input
            y: theano.tensor object used for truths

        Returns: theano function that takes input (train_x,train_y)
                 and returns error and accuracy
        """
        print ("Creating {} Validator...".format(self.name))
        # create prediction
        val_prediction = lasagne.layers.get_output(
            self.network,
            deterministic=True
        )
        # check how much error in prediction
        if self.num_classes is None or self.num_classes == 0:
            val_loss = self.mse_loss(val_prediction, self.y)
        else:
            val_loss = self.cross_entropy_loss(val_prediction, self.y)

        # check the accuracy of the prediction
        val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1),
                              T.argmax(self.y, axis=1)),
                         dtype=theano.config.floatX)

        return theano.function([self.input_var, self.y], [val_loss, val_acc])

    def forward_pass(self, input_data, convert_to_class=False):
        """
        Allow the implementer to quickly get outputs from the network.

        Args:
            input_data: Numpy matrix to make the predictions on
            get_predictions: If the output should return the class
                             with highest probability

        Returns: Numpy matrix with the output probabilities
                 with each class unless otherwise specified.
        """
        if convert_to_class:
            return get_class(self.output(input_data))
        else:
            return self.output(input_data)

    def train(self, epochs, train_x, train_y, val_x, val_y,
              batch_ratio=0.1, plot=True, change_rate=None):
        """
        Train the network.

        Args:
            epochs: how many times to iterate over the training data
            train_x: the training data
            train_y: the training truth
            val_x: the validation data (should not be also in train_x)
            val_y: the validation truth (should not be also in train_y)
            batch_ratio: the percent (0-1) of how much data a batch should have
            plot: boolean if training curves should be plotted while training
            change_rate: a function that takes alpha and returns alpha'

        """
        print ('\nTraining {} in progress...\n'.format(self.name))

        if batch_ratio > 1:
            batch_ratio = 1
        batch_ratio = float(batch_ratio)

        self.record = dict(
            epoch=[],
            train_error=[],
            train_accuracy=[],
            validation_error=[],
            validation_accuracy=[]
        )

        if train_x.shape[0] * batch_ratio < 1.0:
            batch_ratio = 1.0 / train_x.shape[0]
            print ('Warning: Batch ratio too small. Changing to {:.5f}'.format
                   (batch_ratio))
        try:
            for epoch in range(epochs):
                epoch_time = time.time()
                print ("--> Epoch: {} | Epochs left {}".format(
                    epoch,
                    epochs - epoch - 1
                ))

                train_x, train_y = shuffle(train_x, train_y, random_state=0)

                for i in range(int(1 / batch_ratio)):
                    size = train_x.shape[0]
                    b_x = train_x[int(size * (i * batch_ratio)):
                                  int(size * ((i + 1) * batch_ratio))]
                    b_y = train_y[int(size * (i * batch_ratio)):
                                  int(size * ((i + 1) * batch_ratio))]

                    self.trainer(b_x, b_y, self.learning_rate)

                    sys.stdout.flush()
                    sys.stdout.write('\r\tDone {:.1f}% of the epoch'.format
                                     (100 * (i + 1) * batch_ratio))

                    if change_rate is not None:
                        if not callable(change_rate):
                            print ('Parameter change_rate must be a function '
                                   'that returns a new learning rate. '
                                   'Learning rate remains unchanged.')
                            return
                        # print ('Modifying learning rate from {}'.format(
                        #     self.learning_rate)
                        # ),
                        self.learning_rate = change_rate(
                            self.init_learning_rate,
                            self.minibatch_iteration
                        )
                        # print ('to {}'.format(self.learning_rate))
                    self.minibatch_iteration += 1
                train_error, train_accuracy = self.validator(train_x, train_y)
                validation_error, validation_accuracy = self.validator(val_x,
                                                                       val_y)

                self.record['epoch'].append(epoch)
                self.record['train_error'].append(train_error)
                self.record['train_accuracy'].append(train_accuracy)
                self.record['validation_error'].append(validation_error)
                self.record['validation_accuracy'].append(validation_accuracy)
                epoch_time_spent = time.time() - epoch_time
                print ("\n\terror: {} and accuracy: {} in {:.2f}s"
                       .format(
                           train_error,
                           train_accuracy,
                           epoch_time_spent))

                eta = epoch_time_spent * (epochs - epoch - 1)
                minute, second = divmod(eta, 60)
                hour, minute = divmod(minute, 60)
                print ("\tEstimated time left: {}:{}:{} (h:m:s)\n"
                       .format(int(hour), int(minute), int(second)))

                if plot:
                    plt.ion()
                    plt.figure(1)
                    display_record(record=self.record)

        except KeyboardInterrupt:
            print ("\n\n**********Training stopped prematurely.**********\n\n")
        finally:
            self.timestamp = get_timestamp()

    def conduct_test(self, test_x, test_y, figure_path='figures'):
        """Will conduct the test suite to determine model strength."""
        run_test(
            network=self,
            test_x=test_x,
            test_y=test_y,
            figure_path=figure_path
        )

    def __getstate__(self):
        """Pickle save config."""
        pickle_dict = dict()
        for k, v in self.__dict__.items():
            if not issubclass(v.__class__,
                              theano.compile.function_module.Function) \
                and not issubclass(v.__class__,
                                   theano.tensor.TensorVariable):
                pickle_dict[k] = v
        net_parameters = np.array(
            lasagne.layers.get_all_param_values(self.layers,
                                                **{self.name: True})
        )
        if self.input_network is None:
            return (pickle_dict, net_parameters, None, None)
        else:
            pickle_dict['input_network'] = None
            return (pickle_dict,
                    net_parameters,
                    self.input_network['network'].save_name,
                    self.input_network['layer'])

    def __setstate__(self, params):
        """Pickle load config."""
        self.__dict__.update(params[0])
        if params[2] is not None and params[3] is not None:
            input_network = Network.load_model(params[2])
            self.input_var = input_network.input_var
            self.input_network = {'network': input_network,
                                  'layer': params[3]}
        else:
            self.input_var = T.matrix('input')

        self.y = T.matrix('truth')
        self.__init__(self.__dict__['name'],
                      self.__dict__['input_dimensions'],
                      self.__dict__['input_var'],
                      self.__dict__['y'],
                      self.__dict__['units'],
                      self.__dict__['dropouts'],
                      self.__dict__['input_network'],
                      self.__dict__['num_classes'],
                      self.__dict__['activation'],
                      self.__dict__['pred_activation'],
                      self.__dict__['optimizer'],
                      self.__dict__['learning_rate'])
        lasagne.layers.set_all_param_values(self.layers,
                                            params[1],
                                            **{self.name: True})

    def save_model(self, save_path='models'):
        """
        Will save the model parameters to a npz file.

        Args:
            save_path: the location where you want to save the params
        """
        if self.input_network is not None:
            if not hasattr(self.input_network['network'], 'save_name'):
                self.input_network['network'].save_model()

        if not os.path.exists(save_path):
            print ('Path not found, creating {}'.format(save_path))
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "{}{}".format(self.timestamp,
                                                          self.name))
        self.save_name = '{}.network'.format(file_path)
        print ('Saving model as: {}'.format(self.save_name))

        with open(self.save_name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.save_metadata(file_path)

    @classmethod
    def load_model(cls, load_path):
        """
        Will load the model paramters from npz file.

        Args:
            load_path: the exact location where the model has been saved.
        """
        print ('Loading model from: {}'.format(load_path))
        with open(load_path, 'rb') as f:
            instance = pickle.load(f)
        return instance

    def save_record(self, save_path='records'):
        """
        Will save the training records to file to be loaded up later.

        Args:
            save_path: the location where you want to save the records
        """
        if self.record is not None:
            if not os.path.exists(save_path):
                print ('Path not found, creating {}'.format(save_path))
                os.makedirs(save_path)

            file_path = os.path.join(save_path, "{}{}".format(self.timestamp,
                                                              self.name))
            print ('Saving records as: {}_stats.pickle'.format(file_path))
            with open('{}_stats.pickle'.format(file_path), 'w') as output:
                pickle.dump(self.record, output, -1)
        else:
            print ("Error: Nothing to save. Try training the model first.")

    def save_metadata(self, file_path='models'):
        """
        Will save network configuration alongside weights.

        Args:
            file_path: the npz file path without the npz
        """
        config = {
            "{}".format(file_path): {
                "input_dimensions": self.input_dimensions,
                "input_var": "{}".format(self.input_var.type),
                "y": "{}".format(self.y.type),
                "units": self.units,
                "dropouts": self.dropouts,
                "num_classes": self.num_classes,
                "input_network": {
                    'network': None,
                    'layer': None
                }
            }
        }

        if self.input_network:
            config["{}".format(file_path)]["input_network"]['network'] = \
                self.input_network['network'].save_name
            config["{}".format(file_path)]["input_network"]['layer'] = \
                self.input_network['layer']

        json_file = "{}_metadata.json".format(file_path)
        print ('Saving metadata to {}'.format(json_file))
        with open(json_file, 'w') as file:
            json.dump(config, file)

if __name__ == "__main__":
    pass
