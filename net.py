import theano
import theano.tensor as T
import lasagne
import pandas as pd
import numpy as np

def create_dense_network(dimensions, input_var):
    '''
        Generates a fully connected layer
        Args:
            dimension: the size of the incoming theano tensor
            input_var: a theano tensor representing your data input

        Returns: the output of the network (linked up to all the layers)
    '''
    print ("Creating Network...")
    network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)
    print ('Input Layer:')
    print ' ',lasagne.layers.get_output_shape(network)
    print ('Hidden Layer:')

    network = lasagne.layers.DenseLayer(network, 
                                        num_units=32000,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network,p=0.5)
    print ' ',lasagne.layers.get_output_shape(network)

    network = lasagne.layers.DenseLayer(network,
                                        num_units=16000,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network,p=0.5)
    print ' ',lasagne.layers.get_output_shape(network)

    network = lasagne.layers.DenseLayer(network,
                                        num_units=8000,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network,p=0.5)
    print ' ',lasagne.layers.get_output_shape(network)

    network = lasagne.layers.DenseLayer(network,    
                                        num_units=2,
                                        nonlinearity = lasagne.nonlinearities.softmax)
    print ('Output Layer:')
    print ' ',lasagne.layers.get_output_shape(network)

    return network

def main():

    data = pd.read_csv('data/Citalopram_study.csv',low_memory='false',header = None,
                      index_col=0)

    data = data.transpose()
    del data['Response']
    del data['Remission']
    del data['FileGroup']
    del data['Accession Id']
    del data['TimePoint']
    del data['(ng/ml/mg CIT dose)']
    del data['%improvement']
    num_patients = np.count_nonzero(pd.unique(data.values[:,0]))
    num_attributes = np.count_nonzero(pd.unique(data.values[0]))

    import pudb; pu.db
    np_data = np.array(data)
    input_var = T.dmatrix('input')
    y = T.dmatrix('truth')
    network = create_network(np_data.shape,input_var)

if __name__ == "__main__":
    main()