import numpy
import csv

from Layer import Layer
from NeuralNetClassifier import NeuralNetClassifier
from neurons.InnerNeuron import InnerNeuron
from neurons.InputNeuron import InputNeuron
from statistic_functions.Sigmoid import Sigmoid

# Load data
data = numpy.fromfile('../data/train_x.bin', dtype='uint8')
data = data.reshape((100000, 3600))

# Load the results (parallel to data)
results = []
with open('../data/train_y.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                results.append(row)
results = results[1:]
results = [int(entry[1]) for entry in results]

def cross_validate(k_fold, data, results):
    partition_size = int(len(data)/k_fold)
    data_partitions = []
    result_partitions = []
    for i in range(0, k_fold):
        start_ind = i * partition_size
        end_ind = start_ind + partition_size
        if i == (k_fold - 1):
            data_partitions.append(data[start_ind:])
            result_partitions.append(results[start_ind:])
        else:
            data_partitions.append(data[start_ind:end_ind])
            result_partitions.append(results[start_ind:end_ind])

    cross_results = []
    for i in range(0, len(data_partitions)):
        net = NeuralNetClassifier.fully_connected(len(data[0]), 19, [10, 10, 5], Sigmoid())
        net.hidden_layers[-1].add_bias_term(1)
        
        validation_data = data_partitions[i]
        validation_results = result_partitions[i]

        training_data = data_partitions[:i] + data_partitions[i+1:]
        # flatten
        training_data = [item for sublist in training_data for item in sublist]
        
        training_results = result_partitions[:i] + result_partitions[i+1:]
        # flatten
        training_results = [item for sublist in training_results for item in sublist]

        print("Begining training on fold %i..." % i)
        for j in range(0, len(training_data)):
            print("learning example %i" % j)
            inputs = training_data[j]
            outputs = [0] * 19
            outputs[training_results[j]] = 1
            net.learn_example(inputs, outputs, 0.5)

        t_correct = 0
        for j in range(0, len(training_data)):
            prediction = net.predict(training_data[j])
            prediction = prediction.index(max(prediction))
            if prediction == training_results[j]:
                t_correct += 1
        t_correct_per = t_correct/len(training_data)

        v_correct = 0
        for j in range(0, len(validation_data)):
            prediction = net.predict(validation_data[j])
            prediction = prediction.index(max(prediction))
            if prediction == validation_results[j]:
                v_correct += 1
        v_correct_per = v_correct/len(validation_data)

        cross_results.append((t_correct_per, v_correct_per))
    for i in range(0, len(cross_results)):
        print("========Fold %i=========" % i)
        print("training: %f" % cross_results[i][0])
        print("validation: %f" % cross_results[i][1])
