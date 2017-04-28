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
                                        num_units=2048,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network,p=0.5)
    print ' ',lasagne.layers.get_output_shape(network)

    network = lasagne.layers.DenseLayer(network,
                                        num_units=2048,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network,p=0.5)
    print ' ',lasagne.layers.get_output_shape(network)

    network = lasagne.layers.DenseLayer(network,
                                        num_units=1024,
                                        nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DropoutLayer(network,p=0.5)
    print ' ',lasagne.layers.get_output_shape(network)

    network = lasagne.layers.DenseLayer(network,    
                                        num_units=2,
                                        nonlinearity = lasagne.nonlinearities.softmax)
    print ('Output Layer:')
    print ' ',lasagne.layers.get_output_shape(network)

    return network

def get_modified_truth(in_matrix):
    '''
        Reformats truth matrix to be the same size as the output of the dense network
        Args:
            in_matrix: the categorized 1D matrix (dtype needs to be category)

        Returns: a correctly formatted numpy array of the truth matrix
    '''
    temp = np.zeros(shape=(1,len(in_matrix.cat.categories)), dtype='float32')
    for i in np.array(in_matrix.cat.codes):
        row = np.zeros((1,len(in_matrix.cat.categories)))
        row[0,i] = 1.0
        temp = np.concatenate((temp,row),axis=0)
    return np.array(temp[1:],dtype='float32')

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

    data['Gender'] = data['Gender'].astype('category')
    train_y = get_modified_truth(data['Gender'])
    del data['Gender']
    train_id = data['id_response']
    del data['id_response']
    #data['HAMD Score'] = data['HAMD Score'].astype('int32')
    #data['Age'] = data['Age'].astype('int32')
    data = data.astype('float32')
    
    train_X = np.array(data)
    
    input_var = T.dmatrix('input')
    y = T.dmatrix('truth')
    network = create_dense_network(train_X.shape,input_var)
    test_fn = theano.function([input_var], lasagne.layers.get_output(network))

if __name__ == "__main__":
    main()