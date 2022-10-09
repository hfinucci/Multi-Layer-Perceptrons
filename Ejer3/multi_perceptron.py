from Ejer3.layer import Layer
import numpy as np


class MultiPerceptron:
    ERROR_MIN = 0.001

    def __init__(self, net_congif, learn_rate, Activation):
        self.layers = []
        for i in range(1, len(net_congif)):
            self.layers.append(Layer(net_congif[i], net_congif[i - 1], Activation, learn_rate))

        self.num_layers = len(net_congif) - 1

    def get_prev_outputs(self, current_layer, inputs):
        index = current_layer - 1
        if index < 0:
            return inputs
        else:
            return self.layers[current_layer - 1].get_all_outputs()

    def get_expected(self, current_layer, current_neuron, expected_value):
        if current_layer + 1 < self.num_layers and expected_value is None:
            return self.layers[current_layer + 1].get_expected_inner(current_neuron)
        else:
            return expected_value - self.layers[current_layer].neurons[current_neuron].output

    def plot(self):
        for i in range(0, self.num_layers):
            self.layers[i].plot(i)

    def forward_propagation(self, input):
        for i in range(0, self.num_layers):
            self.layers[i].propagation(self.get_prev_outputs(i, input))

    def back_propagation(self, expected_value):
        for layer in range(self.num_layers - 1, 0):
            inputs = self.get_prev_outputs(layer, expected_value)
            for neuron in range(0, self.layers[layer].get_size()):
                expected = self.get_expected(layer, neuron, expected_value)
                self.layers[layer].neurons[neuron].update_w(inputs, expected)

    def train(self, training, expected_output, max_gen):
        current_gen = 0
        error = 0
        self.error_min = np.inf

        # no seria len(training) ?
        positions = np.arange(0, len(self.training))

        while self.error_min > self.ERROR_MIN and current_gen < max_gen:
            np.random.shuffle(positions)
            for i in positions:
                self.forward_propagation(training[i])
                self.back_propagation(expected_output[i])

            #TODO: Calcular error
            error_min = self.calculate_error()

    def test(self, test_set):
        pass
