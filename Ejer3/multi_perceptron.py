from Ejer3.layer import Layer
import numpy as np


class MultiPerceptron:
    ERROR_MIN = 0.01

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

    def calculate_error(self, expected_output):
        m = len(self.layers)
        neurons = self.layers[m - 1].neurons
        aux_sum = 0
        for i in range(len(neurons)):
            aux_sum += (expected_output[i] - neurons[i].output) ** 2
        return aux_sum

    def train(self, training, expected_output, max_gen):
        current_gen = 0
        error = 0
        self.error_min = np.inf

        errors = []
        positions = np.arange(0, len(training))

        while self.error_min > self.ERROR_MIN and current_gen < max_gen:
            np.random.shuffle(positions)
            for i in positions:
                self.forward_propagation(training[i])
                self.back_propagation(expected_output[i])
                
                error = self.calculate_error(expected_output[i])

                if error < self.error_min:
                    self.error_min = error
                    print(self.error_min)
                    errors.append(float(error))
                    if self.error_min < self.ERROR_MIN:
                        print("termine!")
                        return errors

            current_gen += 1

        return errors
            
    def save(self, filepath):
        file = open(filepath, "w+")
        for layer in self.layers:
            for neuron in layer.neurons:
                wcount = 1
                for weight in neuron.weights:
                    file.write("w%d: %d" % wcount % weight)
                    wcount += 1
                file.write("\n")
        print("termine!")

        file.close()


    def test(self, test_set):
        pass
