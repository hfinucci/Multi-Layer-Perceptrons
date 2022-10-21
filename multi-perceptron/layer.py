from neuron import Neuron
import numpy as np


class Layer:

    def __init__(self, num_neurons, prev_num_neurons, activation, learn_rate):
        self.neurons = np.array(list(Neuron(prev_num_neurons, activation, learn_rate, ) for i in range(num_neurons)))

    def get_layer_inputs(self, layer_index, perceptron_inputs):
        index = layer_index - 1
        if index < 0:
            return perceptron_inputs
        else:
            return self.layers[index].get_all_outputs()

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
