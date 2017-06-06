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
    gender_data = get_one_hot(data['Gender'])
    train_id = np.array(data['id_response'])

    del data['Gender']
    del data['id_response']

    data = data.astype('float32')
    data = np.array(data)

    # data = data[:, :5]
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
        y=y,
        units=[4096, 2048, 1024],
        dropouts=[0.2, 0.2, 0.2],
        input_network=None,
        num_classes=2
    )
    # Use to load model from disk
    # dense_net.load_model(load_path='models/3_dense.npz')

    dense_net.train(
        epochs=5,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        batch_ratio=0.25,
        plot=True
    )
    dense_net.conduct_test(test_x=val_x, test_y=val_y)
    # Use to run the test suite on the model
    # dense_net.conduct_test(test_x=val_x, test_y=val_y)

    # Use to save model parameters to disk
    # dense_net.save_model(save_path='models')

    # Use to save training curves to disk
    # dense_net.save_record(save_path='records')

    # Use to get a function to get output of network
    '''
    test_out = dense_net.forward_pass(
        input_data=val_x,
        convert_to_class=True
    )
    '''

if __name__ == "__main__":
    main()
