"""Contains the class for creating networks."""

import time

import numpy as np

import lasagne

import matplotlib.pyplot as plt

import theano
import theano.tensor as T


class Network(object):
    """Class to generate networks and train them."""

    def __init__(self, name, dimensions, input_var, y):
        """
        Initialize network specified.

        Args:
            dimensions: the size of the input data matrix
            input_var: theano tensor representing input matrix
            y: theano tensor representing truth matrix
        """
        self.name = name
        self.dimensions = dimensions
        self.input_var = input_var
        self.y = y
        self.network = self.create_dense_network()
        self.trainer = self.create_trainer()
        self.validator = self.create_validator()
        self.record = None

    def create_dense_network(self):
        """
        Generate a fully connected layer.

        Args:
            dimension: the size of the incoming theano tensor
            input_var: a theano tensor representing your data input

        Returns: the output of the network (linked up to all the layers)
        """
        print ("Creating Network...")
        print ('Input Layer:')
        network = lasagne.layers.InputLayer(shape=self.dimensions,
                                            input_var=self.input_var)
        print ' ', lasagne.layers.get_output_shape(network)

        print ('Hidden Layer:')
        network = lasagne.layers.DenseLayer(network,
                                            num_units=4096,
                                            nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DropoutLayer(network, p=0.5)
        print ' ', lasagne.layers.get_output_shape(network)

        network = lasagne.layers.DenseLayer(network,
                                            num_units=2048,
                                            nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DropoutLayer(network, p=0.5)
        print ' ', lasagne.layers.get_output_shape(network)

        network = lasagne.layers.DenseLayer(network,
                                            num_units=1024,
                                            nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.DropoutLayer(network, p=0.5)
        print ' ', lasagne.layers.get_output_shape(network)

        network = lasagne.layers.DenseLayer(network,
                                            num_units=2,
                                            nonlinearity=lasagne.nonlinearities.softmax)

        print ('Output Layer:')
        print ' ', lasagne.layers.get_output_shape(network)

        return network

    def create_trainer(self):
        """
        Generate a theano function to train the network.

        Args:
            network: Lasagne object representing the network
            input_var: theano.tensor object used for data input
            y: theano.tensor object used for truths

        Returns: theano function that takes as input (train_x,train_y) and trains the net
        """
        print ("Creating Trainer...")
        # get network output
        out = lasagne.layers.get_output(self.network)
        # get all trainable parameters from network
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        # calculate a loss function which has to be a scalar
        cost = T.nnet.categorical_crossentropy(out, self.y).mean()
        # calculate updates using ADAM optimization gradient descent
        updates = lasagne.updates.adam(
            cost,
            params,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        )
        # omitted (, allow_input_downcast=True)
        return theano.function([self.input_var, self.y], updates=updates)

    def create_validator(self):
        """
        Generate a theano function to check the error and accuracy of the network.

        Args:
            network: Lasagne object representing the network
            input_var: theano.tensor object used for data input
            y: theano.tensor object used for truths

        Returns: theano function that takes input (train_x,train_y)
                 and returns error and accuracy
        """
        print ("Creating Validator...")
        # create prediction
        val_prediction = lasagne.layers.get_output(
            self.network,
            deterministic=True
        )
        # check how much error in prediction
        val_loss = lasagne.objectives.categorical_crossentropy(
            val_prediction,
            self.y
        ).mean()
        # check the accuracy of the prediction
        val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), T.argmax(self.y, axis=1)),
                        dtype=theano.config.floatX)

        return theano.function([self.input_var, self.y], [val_loss, val_acc])

    def train(self, epochs, train_x, train_y, val_x, val_y, plot=True):
        """
        Train the network.

        Args:
            epochs: how many times to iterate over the training data
            train_x: the training data
            train_y: the training truth
            val_x: the validation data (should not be also in train_x)
            val_y: the validation truth (should not be also in train_y)
            plot: A boolean if the training curves should be plotted while training

        """
        self.record = dict(
            epoch=[],
            train_error=[],
            train_accuracy=[],
            validation_error=[],
            validation_accuracy=[]
        )
        if plot:
            plt.ion()

        for epoch in range(epochs):
            epoch_time = time.time()
            print ("--> Epoch: %d | Epochs left %d" % (epoch, epochs - epoch))

            self.trainer(train_x, train_y)
            train_error, train_accuracy = self.validator(train_x, train_y)
            validation_error, validation_accuracy = self.validator(val_x, val_y)

            self.record['epoch'].append(epoch)
            self.record['train_error'].append(train_error)
            self.record['train_accuracy'].append(train_accuracy)
            self.record['validation_error'].append(validation_error)
            self.record['validation_accuracy'].append(validation_accuracy)
            print ("    error: %s and accuracy: %s in %.2fs\n" % (
                train_error,
                train_accuracy,
                time.time() - epoch_time)
            )

            if plot:
                plt.plot(
                    self.record['epoch'],
                    self.record['train_error'],
                    '-mo',
                    label='Train Error' if epoch == 0 else ""
                )
                plt.plot(
                    self.record['epoch'],
                    self.record['train_accuracy'],
                    '-go',
                    label='Train Accuracy' if epoch == 0 else ""
                )
                plt.plot(
                    self.record['epoch'],
                    self.record['validation_error'],
                    '-ro',
                    label='Validation Error' if epoch == 0 else ""
                )
                plt.plot(
                    self.record['epoch'],
                    self.record['validation_accuracy'],
                    '-bo',
                    label='Validation Accuracy' if epoch == 0 else ""
                )
                plt.xlabel("Epoch")
                plt.ylabel("Cross entropy error")
                # plt.ylim(0,1)
                plt.title('Training on predicting gender')
                plt.legend(loc='upper right')

                plt.show()
                plt.pause(0.0001)

    def save_model(self, save_path):
        """
        Will save the model parameters to a npz file.

        Args:
            save_path: the location where you want to save the params
        """
        network_name = '%s%s.npz' % (save_path, self.name)
        print ('Saving model as %s' % network_name)
        np.savez(network_name, *lasagne.layers.get_all_param_values(self.network))

    def load_model(self, load_path):
        """
        Will load the model paramters from npz file.

        Args:
            load_path: the exact location where the model has been saved.
        """
        with np.load(load_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values)

    def save_record(self, save_path):
        """
        Will save the training records to file to be loaded up later.

        Args:
            save_path: the location where you want to save the records
        """
        import pickle
        with open('%s_stats.pickle' % self.name, 'w') as output:
            pickle.dump(self.record, output)

if __name__ == "__main__":
    pass
