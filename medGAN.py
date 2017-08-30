"""Contains the train script and data formatting."""

import numpy as np

import pandas as pd

import lasagne

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

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

    srng = RandomStreams(seed=234)
    z = srng.uniform((10, 32))

    generator_config = {
        'mode': 'dense',
        'units': [4096, 2048, 784],
        'dropouts': [0.5, 0.5, 0.5],
    }

    generator = Network(
        name='generator',
        dimensions=(None, 32),
        input_var=z,
        y=None,
        config=generator_config,
        input_network=None,
        num_classes=None
    )

    discriminator_config = {
        'mode': 'dense',
        'units': [4096, 2048, 784],
        'dropouts': [0.5, 0.5, 0.5],
    }

    discriminator = Network(
        name='discriminator',
        dimensions=(None, 784),
        input_var=None,
        y=None,
        config=discriminator_config,
        input_network={'network': generator, 'layer': 6, 'get_params': False},
        num_classes=1,
        pred_activation='sigmoid'
    )

    g_x = lasagne.layers.get_output(generator.layers)[-1]
    score_g_x = lasagne.layers.get_output(discriminator.layers, g_x)[-1]
    score_r_x = lasagne.layers.get_output(discriminator.layers, input_var)[-1]

    discriminator_loss = (score_g_x - score_r_x).mean()
    generator_loss = (- score_g_x).mean()

    discriminator.cost = discriminator_loss
    generator.cost = generator_loss
    discriminator.input_var = input_var
    discriminator.trainer = discriminator.create_trainer()
    generator.trainer = generator.create_trainer()

    import pudb; pu.db
    # Use to load model from disk
    # dense_net.load_model(load_path='models/3_dense.npz')

    discriminator.train(
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
