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

    def __init__(self, name, template_network, n_snapshots, n_epochs,
                 init_learning_rate):
        """
        Initialize snapshot ensemble given a network.

        Args:
            name: string of snapshot ensemble name
            network: Network object which you want to ensemble
            M: number of snapshots in ensemble
            T: total number of epochs
            init_learning_rate: start learning rate
        """
        self.name = name
        self.timestamp = get_timestamp()
        self.template_network = template_network
        self.M = n_snapshots
        self.T = n_epochs
        self.init_learning_rate = init_learning_rate
        if template_network is not None:
            self.template_network.learning_rate = init_learning_rate
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

    def train(self, train_x, train_y, val_x, val_y, batch_ratio, plot):
        """
        Train each model for T/M epochs and sets new network learning rate.

        Collects each model in a class variable self.networks
        """
        for i in range(self.M):

            self.template_network.train(
                epochs=self.T // self.M,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                batch_ratio=batch_ratio,
                plot=plot,
                change_rate=self.cos_annealing
            )
            self.networks += [deepcopy(self.template_network)]
            self.template_network.learning_rate = self.init_learning_rate
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

    def save_ensemble(self, save_path='models'):
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
    def load_ensemble(cls, ensemble_path):
        """Load up ensembled models with a folder location."""
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
            template_network=None,
            n_snapshots=config[ensemble_path]['n_snapshots'],
            n_epochs=config[ensemble_path]['n_epochs'],
            init_learning_rate=config[ensemble_path]['init_learning_rate']
        )
        snap.networks = networks
        return snap

    def save_ensemble_metadata(self, file_path):
        """
        Will save ensemble configuration.

        Args:
            file_path: the npz file path without the npz

            self, name, template_network, n_snapshots, n_epochs,
                 init_learning_rate):
        """
        config = {
            "{}".format(file_path): {
                "n_snapshots": self.M,
                "n_epochs": self.T,
                "init_learning_rate": self.init_learning_rate,
                "networks": [{n.name: n.save_name} for n in self.networks]
            }
        }

        json_file = "{}_metadata.json".format(
            os.path.join(file_path, os.path.basename(file_path))
        )
        print ('Saving metadata to {}'.format(json_file))
        with open(json_file, 'w') as file:
            json.dump(config, file)
