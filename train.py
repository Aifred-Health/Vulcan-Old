import numpy as np

import pandas as pd

import theano.tensor as T

from net import Network


def get_modified_truth(in_matrix):
    """
    Reformat truth matrix to be the same size as the output of the dense network.

    Args:
        in_matrix: the categorized 1D matrix (dtype needs to be category)

    Returns: a correctly formatted numpy array of the truth matrix
    """
    temp = np.zeros(shape=(1, len(in_matrix.cat.categories)), dtype='float32')
    for i in np.array(in_matrix.cat.codes):
        row = np.zeros((1, len(in_matrix.cat.categories)))
        row[0, i] = 1.0
        temp = np.concatenate((temp, row), axis=0)
    return np.array(temp[1:], dtype='float32')


def main():
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
    del data['Gender']
    train_id = np.array(data['id_response'])
    del data['id_response']
    # data['HAMD Score'] = data['HAMD Score'].astype('int32')
    # data['Age'] = data['Age'].astype('int32')
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
        dimensions=(None, int(train_x.shape[1])),
        input_var=input_var,
        y=y
    )

    dense_net.train(
        epochs=100,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        plot=True
    )

    # Use to get a function to get output of network
    # test_fn = theano.function([input_var], lasagne.layers.get_output(network))

if __name__ == "__main__":
    main()
