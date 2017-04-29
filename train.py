"""Contains the train script and data formatting."""

import numpy as np

import pandas as pd

import theano.tensor as T

from net import Network

from utils import get_modified_truth


def main():
    """Open data and format for model training."""
    # how much of the data do you want to reserve for training
    train_reserve = 0.7

    data = pd.read_csv(
        'data/Citalopram_study.csv',
        low_memory='false',
        header=None,
        index_col=0)

    data = data.transpose()
    del data['Response']
    del data['Remission']
    del data['FileGroup']
    del data['Accession Id']
    del data['TimePoint']
    del data['(ng/ml/mg CIT dose)']
    del data['%improvement']
    # num_patients = np.count_nonzero(pd.unique(data.values[:, 0]))
    # num_attributes = np.count_nonzero(pd.unique(data.values[0]))

    data['Gender'] = data['Gender'].astype('category')
    gender_data = get_modified_truth(data['Gender'])
    train_id = np.array(data['id_response'])

    del data['Gender']
    del data['id_response']

    data = data.astype('float32')
    data = np.array(data)

    # Used to shuffle matrices in unison
    permutation = np.random.permutation(data.shape[0])
    data = data[permutation]
    train_id = train_id[permutation]
    gender_data = gender_data[permutation]

    train_x = data[:int(data.shape[0] * train_reserve)]
    val_x = data[int(data.shape[0] * train_reserve):]
    train_y = gender_data[:int(gender_data.shape[0] * train_reserve)]
    val_y = gender_data[int(gender_data.shape[0] * train_reserve):]

    input_var = T.fmatrix('input')
    y = T.fmatrix('truth')

    dense_net = Network(
        name='3_dense',
        dimensions=(None, int(train_x.shape[1])),
        input_var=input_var,
        y=y
    )

    # Use to load model from disk
    # dense_net.load_model(load_path='models/3_dense.npz')

    dense_net.train(
        epochs=10,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        plot=True
    )

    # Use to save model parameters to disk
    # dense_net.save_model(save_path='models')

    # Use to save training curves to disk
    # dense_net.save_record(save_path='records')

    # Use to get a function to get output of network
    # test_fn = theano.function([input_var], lasagne.layers.get_output(network))

if __name__ == "__main__":
    main()
