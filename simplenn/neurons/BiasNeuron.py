from neurons.InputNeuron import InputNeuron

# Neuron to add bias to a Layer
# Must have dummy training methods so recursive calls don't break
class BiasNeuron(InputNeuron):
    def calculate_correction(self, share_of_error):
        # Do nothing
        return

    def update_parent_deltas(self):
        # Do nothing
        return

    def correct_weights(self, step_size):
        # Do nothing
        return

    def add_input(self, neuron):
        # Do nothing
        return
