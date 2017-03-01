from neurons.BiasNeuron import BiasNeuron

class Layer:
    def __init__(self):
        self.neurons = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def fully_connect_with(self, other_layer):
        for neuron in self.neurons:
            neuron.inputs = []
            for inp in other_layer.neurons:
                neuron.add_input(inp)

    def add_bias_term(self, value):
        # Add bias neuron with value 1
        bias = BiasNeuron()
        bias.set_input_value(value)
        self.neurons.append(bias)
