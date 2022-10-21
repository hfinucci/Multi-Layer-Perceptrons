from neuron import Neuron
import numpy as np


class Layer:

    def __init__(self, num_neurons, prev_num_neurons, activation, learn_rate):
        self.neurons = np.array(list(Neuron(prev_num_neurons, activation, learn_rate) for i in range(num_neurons)))
        self.weight_num = num_neurons

    def get_size(self):
        return len(self.neurons)

    # devuelve en una array los output de todas la neurona de esta capa
    def get_outputs(self):
        outputs = []
        for current_neuron in self.neurons:
            outputs.append(current_neuron.output)
        return outputs

    # devuelve un array con los deltas para las neuronas de la layer anterior
    def get_deltas(self):
        delta = [0 for x in range(self.weight_num)]
        for current_neuron in self.neurons:
            for w_index in range(self.weight_num):
                delta[w_index] += (current_neuron.weights[w_index] * current_neuron.error)
        return delta

    # calcula el error para todas las neuronas
    def calculate_error(self, deltas: np.ndarray):
        for i in range(len(self.neurons)):
            self.neurons[i].calculate_error(deltas[i])

    # actualiza el w de todas la neuronas
    def update_w(self, input: np.ndarray):
        for neuron in self.neurons:
            neuron.update_w(input)

    # calcula el output de todas las neuronas
    def calculate_output(self, input):
        for neuron in self.neurons:
            neuron.calculate_output(input)
