"""
An implementation of snapshot ensemble methods.

Described https://arxiv.org/abs/1704.00109. Using the aifred Network API.
"""

import os

import numpy as np

from copy import deepcopy

from utils import get_timestamp
from utils import get_class

from net import Network

import json


class Snapshot(object):
    """Uses Network to build model snapshots."""

    def __init__(self, name, template_network, n_snapshots):
        """
        Initialize snapshot ensemble given a network.

        Args:
            name: string of snapshot ensemble name
            template_network: Network object which you want to ensemble
            n_snapshots: number of snapshots in ensemble
            n_epochs: total number of epochs
            batch_ratio: the minibatch size to total ratio
        """
        self.name = name
        self.timestamp = get_timestamp()
        self.template_network = template_network

        self.num_classes = template_network.num_classes

        if n_snapshots <= 0:
            print('Number of Snapshots too small. Setting to 1.')
            self.M = 1
        else:
            self.M = n_snapshots

        self.networks = []

    def cos_annealing(self, alpha, t):
        """
        Cosine annealing for fast convergence in snapshot learning.

        Args:
            alpha: the old learning rate
            t: current iteration

        Returns new learning rate
        """
        inner_cos = (np.pi * (t % (self.T // self.M))) / (self.T // self.M)
        outer_cos = np.cos(inner_cos) + 1
        return float(alpha / 2 * outer_cos)

    def train(self, epochs, train_x, train_y, val_x, val_y,
              batch_ratio=0.05, plot=True, change_rate=None):
        """
        Train each model for T/M epochs and sets new network learning rate.

        Collects each model in a class variable self.networks
        """
        if epochs < self.M:
            print('Number of epochs to small for number of Snapshots. '
                  'Setting to {}.'.format(self.M))
            self.n_epochs = self.M
        else:
            self.n_epochs = epochs
        self.batch_ratio = batch_ratio
        self.T = self.n_epochs / batch_ratio

        for i in range(self.M):

            self.template_network.train(
                epochs=self.n_epochs // self.M,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                batch_ratio=batch_ratio,
                plot=plot,
                change_rate=self.cos_annealing
            )
            self.networks += [deepcopy(self.template_network)]
        self.timestamp = get_timestamp()

    def forward_pass(self, input_data, m=0, convert_to_class=False):
        """
        Get output of ensemble of the last m networks where m <= n_snapshots.

        Args:
            input_data: Numpy matrix to make the predictions on
            m: the m most recent models from the ensemble to give outputs
               Default to get output from all models.
            convert_to_class: return class predictions from ensemble
        """
        if m < 0 or m > len(self.networks):
            print('Select the m most recent models to get output from. '
                  'Setting m to 0 (default to all models)\n')
            m = 0

        prediction_collection = []
        for net in self.networks[-m:]:
            prediction_collection += [net.forward_pass(input_data=input_data,
                                      convert_to_class=False)]
        prediction_collection = np.array(prediction_collection)
        raw_prediction = np.mean(prediction_collection, axis=0,
                                 dtype='float32')
        if convert_to_class:
            return get_class(raw_prediction)
        else:
            return raw_prediction

    def save_model(self, save_path='models'):
        """Save all ensembled networks in a folder with ensemble name."""
        ensemble_path = "{}{}".format(self.timestamp, self.name)
        new_save_path = os.path.join(save_path, ensemble_path)
        if not os.path.exists(new_save_path):
            print ('Creating {} folder'.format(new_save_path))
            os.makedirs(new_save_path)

        for model in self.networks:
            model.save_model(save_path=new_save_path)

        self.save_ensemble_metadata(new_save_path)

    @classmethod
    def load_model(cls, ensemble_path):
        """Load up ensembled models given a folder location."""
        json_file = "{}_metadata.json".format(
            os.path.join(ensemble_path, os.path.basename(ensemble_path))
        )
        with open(json_file, 'r') as file:
            config = json.load(file)

        networks = []
        for model_file in sorted(os.listdir(ensemble_path)):
            if model_file.endswith('.network'):
                file = os.path.join(ensemble_path, model_file)
                networks += [Network.load_model(file)]

        snap = Snapshot(
            name='snap1',
            template_network=networks[0],
            n_snapshots=config[ensemble_path]['n_snapshots']
        )
        snap.networks = networks
        return snap

    def save_ensemble_metadata(self, file_path):
        """
        Will save ensemble configuration.

        Args:
            file_path: the npz file path without the npz
        """
        config = {
            "{}".format(file_path): {
                "n_snapshots": self.M,
                "init_learning_rate": self.template_network.init_learning_rate,
                "networks": [{n.name: n.save_name} for n in self.networks]
            }
        }

        json_file = "{}_metadata.json".format(
            os.path.join(file_path, os.path.basename(file_path))
        )
        print ('Saving metadata to {}'.format(json_file))
        with open(json_file, 'w') as file:
            json.dump(config, file)

    def save_record(self, save_path='records'):
        """Save individual training curves for all networks."""
        ensemble_path = "{}{}".format(self.timestamp, self.name)
        new_save_path = os.path.join(save_path, ensemble_path)
        if not os.path.exists(new_save_path):
            print ('Creating {} folder'.format(new_save_path))
            os.makedirs(new_save_path)

        for model in self.networks:
            model.save_record(save_path=new_save_path)
