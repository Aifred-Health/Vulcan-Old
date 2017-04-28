import theano
import theano.tensor as T
import lasagne
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

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

def create_trainer(network,input_var,y):
    '''
        Generates a theano function to train the network
        Args:
            network: Lasagne object representing the network
            input_var: theano.tensor object used for data input
            y: theano.tensor object used for truths

        Returns: theano function that takes as input (train_X,train_y) and trains the net
    '''
    print ("Creating Trainer...")
    out = lasagne.layers.get_output(network)                        #get network output
    params = lasagne.layers.get_all_params(network, trainable=True) #get all parameters from network
    cost = T.nnet.categorical_crossentropy(out, y).mean()           #calculate a loss function which has to be a scalar
    updates = lasagne.updates.adam(cost,
                                    params,
                                    learning_rate=0.001,
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1e-08)                  #calculate updates using ADAM optimization gradient descent

    
    return theano.function([input_var, y], updates=updates)         # omitted (, allow_input_downcast=True)

def create_validator(network, input_var, y):
    '''
        Generates a theano function to check the error and accuracy of the network
        Args:
            network: Lasagne object representing the network
            input_var: theano.tensor object used for data input
            y: theano.tensor object used for truths

        Returns: theano function that takes input (train_X,train_y) and returns error and accuracy
    '''
    print ("Creating Validator...")
    val_prediction = lasagne.layers.get_output(network, deterministic=True)         #create prediction
    val_loss = lasagne.objectives.categorical_crossentropy(val_prediction,y).mean()   #check how much error in prediction
    val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)    #check the accuracy of the prediction

    return theano.function([input_var, y], [val_loss, val_acc])    #check for error and accuracy percentage

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
    train_reserve = 0.7
    validation_reserve = 1-train_reserve
    epochs = 100
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
    gender_data = get_modified_truth(data['Gender'])
    del data['Gender']
    train_id = data['id_response']
    del data['id_response']
    #data['HAMD Score'] = data['HAMD Score'].astype('int32')
    #data['Age'] = data['Age'].astype('int32')
    data = data.astype('float32')
    data = np.array(data)

    train_X = data[:int(data.shape[0]*train_reserve)]
    val_X = data[int(data.shape[0]*train_reserve):]
    train_y = gender_data[:int(gender_data.shape[0]*train_reserve)]
    val_y = gender_data[int(gender_data.shape[0]*train_reserve):]
    
    input_var = T.fmatrix('input')
    y = T.fmatrix('truth')
    network = create_dense_network((None,int(train_X.shape[1])),input_var)
    trainer = create_trainer(network,input_var,y)
    validator = create_validator(network,input_var,y)

    record = dict(epoch=[],
                    train_error=[],
                    train_accuracy=[],
                    validation_error=[],
                    validation_accuracy=[])
    plt.ion()
    for epoch in range(epochs):
        epoch_time = time.time()
        print ("--> Epoch: %d | Epochs left %d"%(epoch,epochs-epoch))
        trainer(train_X,train_y)
        train_error, train_accuracy = validator(train_X,train_y)
        validation_error,validation_accuracy = validator(val_X,val_y)
        record['epoch'].append(epoch)
        record['train_error'].append(train_error)
        record['train_accuracy'].append(train_accuracy)
        record['validation_error'].append(validation_error)
        record['validation_accuracy'].append(validation_accuracy)
        print ("    error: %s and accuracy: %s in %.2fs\n"%(train_error,train_accuracy,time.time()-epoch_time))

        plt.plot(record['epoch'],record['train_error'], '-mo',label='Train Error' if epoch == 0 else "")
        plt.plot(record['epoch'],record['train_accuracy'],'-go',label='Train Accuracy' if epoch == 0 else "")
        plt.plot(record['epoch'],record['validation_error'], '-ro',label='Validation Error' if epoch == 0 else "")
        plt.plot(record['epoch'],record['validation_accuracy'],'-bo',label='Validation Accuracy' if epoch == 0 else "")
        plt.xlabel("Epoch")
        plt.ylabel("Cross entropy error")
        #plt.ylim(0,1)
        plt.title('Training on predicting gender')
        plt.legend(loc='upper right')

        plt.show()
        plt.pause(0.0001)

    #test_fn = theano.function([input_var], lasagne.layers.get_output(network))

if __name__ == "__main__":
    main()