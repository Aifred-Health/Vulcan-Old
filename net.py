import theano
import lasagne
import pandas as pd
import numpy as np

def create_network(dimensions, input_var):
    #dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer
    #We have to specify the input size because of the dense layer
    print ("Creating Network...")
    network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)
    print ('Input Layer:')
    print ' ',lasagne.layers.get_output_shape(network)
    print ('Hidden Layer:')

    #extra layers if learning capacity is not reached. e.g the data is high-dimensional

    network = lasagne.layers.DenseLayer(network, num_units=1024, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network,p=0.5)
    print ' ',lasagne.layers.get_output_shape(network)

    network = lasagne.layers.DenseLayer(network, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network,p=0.5)
    print ' ',lasagne.layers.get_output_shape(network)

    network = lasagne.layers.DenseLayer(network, num_units=2048, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network,p=0.5)
    print ' ',lasagne.layers.get_output_shape(network)
    #extra layers if learning capacity is not reached. e.g the data is high-dimensional

    network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)
    print ('Output Layer:')
    print ' ',lasagne.layers.get_output_shape(network)

    return network

def main():

    data = pd.read_csv('Citalopram_study.csv',low_memory='false',header = None,
                      index_col=0)
    #data = np.array(data)
    data = data.transpose()
    del data['Response']
    del data['Remission']
    del data['FileGroup']
    del data['Accession Id']
    del data['TimePoint']
    del data['(ng/ml/mg CIT dose)']
    del data['%improvement']
    numPatients = np.count_nonzero(pd.unique(data.values[:,0]))
    numAttributes = np.count_nonzero(pd.unique(data.values[0]))


    input_var = T.tensor4('input')
    y = T.dmatrix('truth')
    network = create_network()