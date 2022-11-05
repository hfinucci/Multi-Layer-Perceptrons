from multiperceptron.layer import Layer
import numpy as np


class MultiPerceptron:

    def __init__(self, net_config, learn_rate, activation):
        self.layers = np.array(
            list(Layer(net_config[i], net_config[i - 1], activation, learn_rate) for i in range(1, len(net_config))))

    def get_layer_inputs(self, layer_index, perceptron_inputs):
        index = layer_index - 1
        if index < 0:
            return perceptron_inputs
        else:
            return self.layers[index].get_outputs()

    def get_layer_deltas(self, layer_index, perceptron_expected_values):
        if layer_index + 1 < len(self.layers):
            return self.layers[layer_index + 1].get_deltas()
        else:
            outputs = self.layers[layer_index].get_outputs()
            for i in range(len(outputs)):
                outputs[i] = perceptron_expected_values[i] - outputs[i]
            return outputs

    # propaga un input por toda la red
    def forward_propagation(self, perceptron_inputs):
        layer_index = 0
        for layer in self.layers:
            layer.calculate_output(self.get_layer_inputs(layer_index, perceptron_inputs))
            layer_index += 1

    # propaga el error y solo si update_flag = true actualiza las w
    def back_propagation(self, data, expected_value, update_flag):
        for layer_index in range(len(self.layers) - 1, -1, -1):
            inputs = self.get_layer_inputs(layer_index, data)
            deltas = self.get_layer_deltas(layer_index, expected_value)

            self.layers[layer_index].calculate_error(deltas)
            if update_flag:
                self.layers[layer_index].update_w(inputs)

    # revisar
    def calculate_error(self, expected_output):
        m = len(self.layers)
        neurons = self.layers[m - 1].neurons
        aux_sum = 0
        for i in range(len(neurons)):
            aux_sum += (expected_output[i] - neurons[i].output) ** 2
        return aux_sum

    def train(self, training, expected_output, max_gen, error_min, n):
        current_gen = 0
        error = np.inf
        n_samples = 0
        errors = []
        positions = np.arange(0, len(training))

        while error > error_min and current_gen < max_gen:
            np.random.shuffle(positions)
            for i in positions:
                self.forward_propagation(training[i])

                if n_samples == n:
                    update_flag = True
                    n_samples = 0
                else:
                    update_flag = False
                    n_samples = n_samples + 1
                self.back_propagation(training[i], expected_output[i], update_flag)

                error = self.calculate_error(expected_output[i])
                if error < error_min:
                    errors.append(float(error))

            current_gen += 1
        return errors

    def test(self, test_set):
        to_return = []
        for data in test_set:
            self.forward_propagation(data)
            to_return.append(self.layers[-1].get_all_outputs())
        return to_return
