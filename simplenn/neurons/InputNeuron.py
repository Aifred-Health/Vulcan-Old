# Simple neuron to relay input
class InputNeuron:
    def __init__(self):
        self.input_value = 0
        self.last_prediction = 0

    def set_input_value(self, value):
        self.last_prediction = value
        self.input_value = value

    def predict(self):
        return self.input_value
