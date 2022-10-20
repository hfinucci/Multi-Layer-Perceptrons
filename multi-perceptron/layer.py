from neuron import Neuron
import numpy as np


class Layer:

    def __init__(self, num_neurons, prev_num_neurons, activation, learn_rate, max_index, index):
        self.neurons = np.array(list(Neuron(prev_num_neurons, activation, learn_rate, ) for i in range(num_neurons)))
        self.max_index = max_index
        self.index = index


    def get_all_outputs(self):
        outputs = []
        for current_neuron in self.neurons:
            outputs.append(current_neuron.output)
        return outputs

    def get_neuron_delta(self, num_neuron):
        delta = 0
        for current_neuron in self.neurons:
            delta += (current_neuron.weights[num_neuron] * current_neuron.delta)
        return delta

    def propagation(self, input):
        for neuron in self.neurons:
            neuron.calculate_output(input)


    def update
