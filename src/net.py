"""Contains the class for creating networks."""

import time

import os

import numpy as np

import lasagne

import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from utils import get_class

from scipy import integrate


class Network(object):
    """Class to generate networks and train them."""

    def __init__(self, name, dimensions, input_var, y,
                 units, dropouts, input_network=None, num_classes=None):
        """
        Initialize network specified.

        Args:
            dimensions: the size of the input data matrix
            input_var: theano tensor representing input matrix
            y: theano tensor representing truth matrix
            units: The list of number of nodes to have at each layer
            dropout: The list of dropout probabilities to have at each layer
            input_network: append networks together (must be of type Network)
            num_classes: None or int. how many classes to predict
        """
        self.name = name
        self.dimensions = dimensions
        self.input_var = input_var
        self.y = y
        self.input_network = input_network
        self.network = self.create_dense_network(
            units=units,
            dropouts=dropouts
        )
        self.num_classes = num_classes
        if num_classes is not None and num_classes != 0:
            self.network = self.create_classification_layer(
                self.network,
                num_classes=num_classes
            )

        self.trainer = self.create_trainer()
        self.validator = self.create_validator()
        self.output = theano.function(
            [self.input_var],
            lasagne.layers.get_output(self.network))
        self.record = None

    def create_dense_network(self, units, dropouts):
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

        print ("Creating %s Network..." % self.name)
        if self.input_network is None:
            print ('\tInput Layer:')
            network = lasagne.layers.InputLayer(shape=self.dimensions,
                                                input_var=self.input_var)
            print '\t\t', lasagne.layers.get_output_shape(network)
        else:
            network = self.input_network.network
            print ('Appending %s to %s.' % (self.name,
                                            self.input_network.name))

        print ('\tHidden Layer:')
        for (num_units, prob_dropout) in zip(units, dropouts):
            network = lasagne.layers.DenseLayer(
                incoming=network,
                num_units=num_units,
                nonlinearity=lasagne.nonlinearities.rectify
            )
            network = lasagne.layers.DropoutLayer(
                network,
                p=prob_dropout
            )
            print '\t\t', lasagne.layers.get_output_shape(network)

        return network

    def create_classification_layer(self, network, num_classes):
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
            nonlinearity=lasagne.nonlinearities.softmax
        )
        print '\t\t', lasagne.layers.get_output_shape(network)

        return network

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
        print ("Creating %s Trainer..." % self.name)
        # get network output
        out = lasagne.layers.get_output(self.network)
        # get all trainable parameters from network
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        # calculate a loss function which has to be a scalar
        cost = T.nnet.categorical_crossentropy(out, self.y).mean()
        # calculate updates using ADAM optimization gradient descent
        updates = lasagne.updates.adam(
            loss_or_grads=cost,
            params=params,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        )
        # omitted (, allow_input_downcast=True)
        return theano.function([self.input_var, self.y], updates=updates)

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
        print ("Creating %s Validator..." % self.name)
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

    def train(self, epochs, train_x, train_y, val_x, val_y, plot=True):
        """
        Train the network.

        Args:
            epochs: how many times to iterate over the training data
            train_x: the training data
            train_y: the training truth
            val_x: the validation data (should not be also in train_x)
            val_y: the validation truth (should not be also in train_y)
            plot: boolean if training curves should be plotted while training

        """
        print ('\nTraining %s in progress...\n' % self.name)

        self.record = dict(
            epoch=[],
            train_error=[],
            train_accuracy=[],
            validation_error=[],
            validation_accuracy=[]
        )
        if plot:
            plt.ion()
            plt.figure(1)

        for epoch in range(epochs):
            epoch_time = time.time()
            print ("--> Epoch: %d | Epochs left %d" % (
                epoch,
                epochs - epoch - 1
            ))

            self.trainer(train_x, train_y)
            train_error, train_accuracy = self.validator(train_x, train_y)
            validation_error, validation_accuracy = self.validator(val_x,
                                                                   val_y)

            self.record['epoch'].append(epoch)
            self.record['train_error'].append(train_error)
            self.record['train_accuracy'].append(train_accuracy)
            self.record['validation_error'].append(validation_error)
            self.record['validation_accuracy'].append(validation_accuracy)
            epoch_time_spent = time.time() - epoch_time
            print ("    error: %s and accuracy: %s in %.2fs" % (
                train_error,
                train_accuracy,
                epoch_time_spent)
            )
            eta = epoch_time_spent * (epochs - epoch - 1)
            minute, second = divmod(eta, 60)
            hour, minute = divmod(minute, 60)
            print ("    ETA: %d:%02d:%02d (h:m:s)\n" % (hour, minute, second))

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
                plt.title('Training curve for model: %s' % self.name)
                plt.legend(loc='upper right')

                plt.show()
                plt.pause(0.0001)

    def conduct_test(self, test_x, test_y):
        """
        Will conduct the test suite to determine model strength.

        Args:
            test_x: data the model has not yet seen to predict
            test_y: corresponding truth vectors
        """
        if self.num_classes is None or self.num_classes == 0:
            print ('Cannot conduct test: there\'s no classification layer')
            return

        if test_y.shape[1] > 1:
            test_y = get_class(test_y)  # Y is in one hot representation

        raw_prediction = self.forward_pass(input_data=test_x,
                                           convert_to_class=False)
        threshold = 0.0
        x = y = np.array([])
        while (threshold < 1.0):
            prediction = np.where(raw_prediction > threshold, 1.0, 0.0)
            prediction = get_class(prediction)

            tp = float(np.sum(np.logical_and(prediction == 1.0,
                                             test_y == 1.0)))
            tn = float(np.sum(np.logical_and(prediction == 0.0,
                                             test_y == 0.0)))
            fp = float(np.sum(np.logical_and(prediction == 1.0,
                                             test_y == 0.0)))
            fn = float(np.sum(np.logical_and(prediction == 0.0,
                                             test_y == 1.0)))

            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            x = np.append(x, 1.0 - specificity)
            y = np.append(y, sensitivity)

            if abs(threshold - 0.5) < 1e-08:
                accuracy = (tp + tn) / (tp + fp + tn + fn)
                sens = tp / (tp + fn) if tp > 0 else 0.0
                spec = tn / (tn + fp) if tn > 0 else 0.0
                dice = 2 * tp / (2 * tp + fp + fn)
                ppv = tp / (tp + fp) if tp > 0 else 0.0
                npv = tn / (tn + fn) if tn > 0 else 0.0

                print ('%s test\'s results' % self.name)

                print ('\tTP: %i, FP: %i, TN: %i, FN: %i' % (tp, fp, tn, fn))
                print ('\tAccuracy: %.4f' % accuracy)
                print ('\tSensitivity: %.4f' % sens)
                print ('\tSpecificity: %.4f' % spec)
                print ('\tDICE: %.4f' % dice)
                print ('\tPositive Predictive Value: %.4f' % ppv)
                print ('\tNegative Predictive Value: %.4f' % npv)

            threshold += 0.01

        auc = integrate.trapz(y, x)  # NEEDS REPAIR

        print ('\tGenerating ROC ...')
        plt.figure(2)
        plt.ion()
        plt.plot(x, y, label=("AUC: %.4f" % auc))
        plt.title("ROC Curve for %s" % self.name)
        plt.xlabel('1 - specificity')
        plt.ylabel('sensitivity')
        plt.legend(loc='lower right')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.show()

        if not os.path.exists('figures'):
            print ('Creating figures folder')
            os.makedirs('figures')
        print ('\tSaving figure to file: figures/%s_ROC.png' % self.name)
        plt.savefig('figures/%s_ROC.png' % self.name)

    def save_model(self, save_path):
        """
        Will save the model parameters to a npz file.

        Args:
            save_path: the location where you want to save the params
        """
        if not os.path.exists(save_path):
            print ('Path not found, creating %s' % save_path)
            os.makedirs(save_path)
        file_path = os.path.join(save_path, self.name)
        network_name = '%s.npz' % (file_path)
        print ('Saving model as: %s' % network_name)
        np.savez(network_name,
                 *lasagne.layers.get_all_param_values(self.network))

    def load_model(self, load_path):
        """
        Will load the model paramters from npz file.

        Args:
            load_path: the exact location where the model has been saved.
        """
        print ('Loading model from: %s' % load_path)
        with np.load(load_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values)

    def save_record(self, save_path):
        """
        Will save the training records to file to be loaded up later.

        Args:
            save_path: the location where you want to save the records
        """
        if self.record is not None:
            import pickle
            if not os.path.exists(save_path):
                print ('Path not found, creating %s' % save_path)
                os.makedirs(save_path)

            file_path = os.path.join(save_path, self.name)
            print ('Saving records as: %s_stats.pickle' % file_path)
            with open('%s_stats.pickle' % file_path, 'w') as output:
                pickle.dump(self.record, output, -1)
        else:
            print ("Error: Nothing to save. Try training the model first.")

if __name__ == "__main__":
    pass
