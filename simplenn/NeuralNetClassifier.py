from Layer import Layer
from neurons.InnerNeuron import InnerNeuron
from neurons.InputNeuron import InputNeuron
from statistic_functions.Sigmoid import Sigmoid

class NeuralNetClassifier:
    def __init__(self, input_space_dim, output_space_dim):
        self.input_space_dim = input_space_dim
        self.output_space_dim = output_space_dim
        self.hidden_layers = []
        self.input_layer = self.build_input_layer()
        self.output_layer = self.build_output_layer()

    # Alternate constructor to make a fully connected neural net automatically.
    # layers is an array with the number of neurons in each layer
    # i.e. [3,4,2] will make a 3 hidden layer network with 3, 4 and 2 neurons
    # respectively
    @classmethod
    def fully_connected(cls, input_space_dim, output_space_dim, layers, statistic_function):
        net = cls(input_space_dim, output_space_dim)
        # Build layers
        for num in layers:
            layer = Layer()
            for j in range(0, num):
                layer.add_neuron(InnerNeuron(statistic_function))
            net.add_hidden_layer(layer)

        # Fully connect hidden layers
        for i in range(1, len(net.hidden_layers)):
            net.hidden_layers[i].fully_connect_with(net.hidden_layers[i-1])

        return net

    def add_hidden_layer(self, layer):
        self.hidden_layers.append(layer)
        self.hidden_layers[0].fully_connect_with(self.input_layer)
        self.output_layer.fully_connect_with(self.hidden_layers[-1])

    # Hidden layers numbered 1 to x
    def fully_connect_hidden_layer(self, hidden_layer_number):
        index = hidden_layer_number - 1
        self.hidden_layers[index].fully_connect_with(self.hidden_layers[index - 1])

    def build_input_layer(self):
        layer = Layer()
        for i in range(0, self.input_space_dim):
            layer.add_neuron(InputNeuron())
        return layer

    def build_output_layer(self):
        layer = Layer()
        sigmoid = Sigmoid()
        for i in range(0, self.output_space_dim):
            layer.add_neuron(InnerNeuron(sigmoid))
        return layer

    def predict(self, inputs):
        if len(inputs) != self.input_space_dim:
            print("Input size must match input space dimensions")
        else:
            for i in range(0, len(inputs)):
                self.input_layer.neurons[i].set_input_value(inputs[i])
            for layer in self.hidden_layers:
                for neuron in layer.neurons:
                    neuron.predict()
            return [neuron.predict() for neuron in self.output_layer.neurons]

    def learn_example(self, inputs, outputs, step_size=.01):
        if len(inputs) != self.input_space_dim:
            print("Input size must match input space dimensions")
        elif len(outputs) != self.output_space_dim:
            print("Output size must match output space dimensions")
        else:
            # Update the output layer deltas and calculate deltas for parent neurons
            predictions = self.predict(inputs)
            for i in range(0, len(outputs)):
                error = predictions[i] - outputs[i]
                self.output_layer.neurons[i].calculate_correction(error)
                self.output_layer.neurons[i].update_parent_deltas()

            # Tell each layer to update their parent neurons' delta starting
            # from the last hidden layer to the second (since first can't
            # update it's parent which is the input layer)
            for layer in reversed(self.hidden_layers[1:]):
                for neuron in layer.neurons:
                    neuron.update_parent_deltas()

            # Tell each layer to correct their weights from output to first layer
            for neuron in self.output_layer.neurons:
                neuron.correct_weights(step_size)
            for layer in reversed(self.hidden_layers):
                for neuron in layer.neurons:
                    neuron.correct_weights(step_size)
