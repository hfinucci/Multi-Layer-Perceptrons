from layer import Layer
import numpy as np


class MultiPerceptron:

    def __init__(self, net_config, learn_rate, activation):
        max_num_layers = len(net_config) - 1
        self.layers = np.array(
            list(Layer(net_config[i], net_config[i - 1], activation, learn_rate, max_num_layers, i) for i in
                 range(1, len(net_config))))

    def get_layer_inputs(self, layer_index, perceptron_inputs):
        index = layer_index - 1
        if index < 0:
            return perceptron_inputs
        else:
            return self.layers[index].get_all_outputs()

    def get_neuron_error(self, layer_index, neuron_index, perceptron_expected_values):
        if layer_index + 1 < len(self.layers):
            return self.layers[layer_index + 1].get_neuron_delta(neuron_index)
        else:
            neuron = self.layers[layer_index].neurons[neuron_index]
            return perceptron_expected_values[neuron_index] - neuron.output

    def forward_propagation(self, input):
        layer_index = 0
        for layer in self.layers:
            layer.propagation(self.get_layer_inputs(layer_index, input))
            layer_index += 1

    def back_propagation(self, data, expected_value, ):
        for layer_index in range(len(self.layers) - 1, -1, -1):
            inputs = self.get_inputs(layer_index, data)
            neuron_index = 0
            for neuron in self.layers[layer_index].neurons:
                neuron_error = self.get_neuron_error(layer_index, neuron_index, expected_value)
                neuron.update_w(inputs, neuron_error)
                neuron_index += 1

    def calculate_error(self, expected_output):
        m = len(self.layers)
        neurons = self.layers[m - 1].neurons
        aux_sum = 0
        for i in range(len(neurons)):
            aux_sum += (expected_output[i] - neurons[i].output) ** 2
        return aux_sum

    def train(self, training, expected_output, max_gen, ERROR_MIN):
        current_gen = 0
        error = 0
        self.error_min = np.inf

        errors = []
        positions = np.arange(0, len(training))

        while self.error_min > ERROR_MIN and current_gen < max_gen:
            np.random.shuffle(positions)
            for i in positions:
                self.forward_propagation(training[i])
                self.back_propagation(training[i], expected_output[i])

                error = self.calculate_error(expected_output[i])
                if error < self.error_min:
                    self.error_min = error
                    # print(self.error_min)
                    errors.append(float(error))
                    if self.error_min < self.ERROR_MIN:
                        print("termine en %d generaciones" % current_gen)
                        print("error de %d" % self.error_min)
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
        to_return = []
        for data in test_set:
            self.forward_propagation(data)
            to_return.append(self.layers[-1].get_all_outputs())
        return to_return
