"""Contains the train script and data formatting."""

import numpy as np

import pandas as pd

import theano.tensor as T

from src.net import Network

from src.utils import get_one_hot


def main():
    """Open data and format for model training."""
    # how much of the data do you want to reserve for training
    train_reserve = 0.7

    data = pd.read_csv(
        'data/Duloxetine_study.csv',
        low_memory='false',
        header=None,
        index_col=0)

    data = data.transpose()

    del data['response']
    del data['remission']
    del data['ID_complet']
    del data['num']
    del data['id partiel']
    del data['id ']

    # del data['race']
    del data['%']
    del data['study']
    del data['RIN']

    # num_patients = np.count_nonzero(pd.unique(data.values[:, 0]))
    # num_attributes = np.count_nonzero(pd.unique(data.values[0]))

    # Turn categorical data into numerical forms
    data['race'] = data['race'].astype('category')
    data['race'] = data['race'].cat.codes

    data['Sex'] = data['Sex'].astype('category')
    gender_data = get_one_hot(data['Sex'])
    train_id = np.array(data['ID'])

    del data['Sex']
    del data['ID']

    # Turn the height and weight values to decimals and remove the commas
    data['taille'] = [x.replace(',', '.') for x in data['taille']]
    data['poids'] = [x.replace(',', '.') for x in data['poids']]

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
    '''
    test_out = dense_net.forward_pass(
        input_data=val_x,
        convert_to_class=False
    )
    '''

if __name__ == "__main__":
    main()
