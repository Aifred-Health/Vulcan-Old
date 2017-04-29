import time

import lasagne

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import theano
import theano.tensor as T


def create_dense_network(dimensions, input_var):
    """
    Generate a fully connected layer.

    Args:
        dimension: the size of the incoming theano tensor
        input_var: a theano tensor representing your data input

    Returns: the output of the network (linked up to all the layers)
    """
    print ("Creating Network...")
    network = lasagne.layers.InputLayer(shape=dimensions, input_var=input_var)
    print ('Input Layer:')
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


def create_trainer(network, input_var, y):
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
    out = lasagne.layers.get_output(network)
    # get all trainable parameters from network
    params = lasagne.layers.get_all_params(network, trainable=True)
    # calculate a loss function which has to be a scalar
    cost = T.nnet.categorical_crossentropy(out, y).mean()
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
    return theano.function([input_var, y], updates=updates)


def create_validator(network, input_var, y):
    """
    Generate a theano function to check the error and accuracy of the network.

    Args:
        network: Lasagne object representing the network
        input_var: theano.tensor object used for data input
        y: theano.tensor object used for truths

    Returns: theano function that takes input (train_x,train_y) and returns error and accuracy
    """
    print ("Creating Validator...")
    # create prediction
    val_prediction = lasagne.layers.get_output(network, deterministic=True)
    # check how much error in prediction
    val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, y).mean()
    # check the accuracy of the prediction
    val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), T.argmax(y, axis=1)),
                    dtype=theano.config.floatX)

    return theano.function([input_var, y], [val_loss, val_acc])


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
    train_reserve = 0.7
    epochs = 100
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
    network = create_dense_network((None, int(train_x.shape[1])), input_var)
    trainer = create_trainer(network, input_var, y)
    validator = create_validator(network, input_var, y)

    record = dict(
        epoch=[],
        train_error=[],
        train_accuracy=[],
        validation_error=[],
        validation_accuracy=[]
    )
    plt.ion()
    for epoch in range(epochs):
        epoch_time = time.time()
        print ("--> Epoch: %d | Epochs left %d" % (epoch, epochs - epoch))

        trainer(train_x, train_y)
        train_error, train_accuracy = validator(train_x, train_y)
        validation_error, validation_accuracy = validator(val_x, val_y)
        record['epoch'].append(epoch)
        record['train_error'].append(train_error)
        record['train_accuracy'].append(train_accuracy)
        record['validation_error'].append(validation_error)
        record['validation_accuracy'].append(validation_accuracy)
        print ("    error: %s and accuracy: %s in %.2fs\n" % (train_error,
                                                            train_accuracy,
                                                            time.time() - epoch_time))

        plt.plot(
            record['epoch'],
            record['train_error'],
            '-mo',
            label='Train Error' if epoch == 0 else ""
        )
        plt.plot(
            record['epoch'],
            record['train_accuracy'],
            '-go',
            label='Train Accuracy' if epoch == 0 else ""
        )
        plt.plot(
            record['epoch'],
            record['validation_error'],
            '-ro',
            label='Validation Error' if epoch == 0 else ""
        )
        plt.plot(
            record['epoch'],
            record['validation_accuracy'],
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
    # Use to get a function to get output of network
    # test_fn = theano.function([input_var], lasagne.layers.get_output(network))

if __name__ == "__main__":
    main()
