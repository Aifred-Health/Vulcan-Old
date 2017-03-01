import random

class InnerNeuron:
    def __init__(self, statistic_function):
        self.inputs = []
        self.statistic_function = statistic_function
        # Instead of calling predict() recursively, we optimize by storing
        # last predicted value. on a 10000x5 input, 2 layer this saved 30s.
        self.last_prediction = 0.5
        self.delta = 0

    def add_input(self, neuron):
        self.inputs.append(self.Input(neuron))

    def predict(self):
        raw = sum([inp.weight * inp.neuron.last_prediction for inp in self.inputs])
        final = self.statistic_function.apply(raw)
        self.last_prediction = final
        return final

    # Calculates the correction from a given child node
    # The child node calls this method, passing the amount of it's
    # error that is relevent to this parent node
    # for inner nodes this is always weight of parent node * child correction
    # for output nodes this is always the error of the output
    def calculate_correction(self, share_of_error):
        output = self.predict()
        self.delta += share_of_error * self.statistic_function.derivative(output)

    # Calculate my portion of the delta for my parent nodes
    # Passes them their share of my correction
    def update_parent_deltas(self):
        for inp in self.inputs:
            inp.neuron.calculate_correction(self.delta * inp.weight)

    # Update all the weights and then reset delta
    def correct_weights(self, step_size):
        for inp in self.inputs:
            inp.weight += step_size * self.delta * inp.neuron.last_prediction
        self.delta = 0

    class Input:
        def __init__(self, neuron):
            self.neuron = neuron
            self.weight = self.default_weight()
            self.correction = 0

        # Method used to calculate default weights
        # By default it calculates a random number 0-1
        def default_weight(self):
            return random.random()
