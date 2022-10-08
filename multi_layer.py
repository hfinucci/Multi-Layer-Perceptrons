import numpy as np


class Activation:
    # TODO
    @staticmethod
    def sigmoid(x):
        if -700 < x < 700:
            return np.exp(x) / (1 + np.exp(x))
        return 0 if x < 0 else 1

    @staticmethod
    def sigmoid_dx(x):
        # se hace 0 despues de este valor
        if -355 < x < 355:
            return np.exp(x) / np.power(np.exp(x) + 1, 2)
        return 0

    @staticmethod
    def tanh(excitation):
        return np.tanh(excitation)

    @staticmethod
    def tanh_dx(excitation):
        return 1 - np.tanh(excitation) ** 2


class Neuron:
    def __init__(self, weight_num, activation, learn_rate):
        self.weights = np.random.uniform(-1, 1, weight_num)
        self.delta = 0
        self.output = 0
        self.output_dx = 0
        self.activation = activation
        self.learn_rate = learn_rate

    def calculate_output(self, input):
        e = np.inner(input, self.weights)
        self.output = self.activation.sigmoid(e)
        self.output_dx = self.activation.sigmoid_dx(e)

    def update_w(self, inputs, expected):
        inputs.append(1)
        self.delta = self.output * (1 - self.output) * expected
        for i in range(1, len(self.weights)):
            self.weights[i] += self.weights[i] * learn_rate * delta * inputs[i]

    def plot(self):
        print(self.weights)


class Layer:

    def __init__(self, num_neurons, prev_num_neurons, activation, learn_rate):
        self.neurons = [Neuron(prev_num_neurons + 1, activation, learn_rate) for i in range(num_neurons)]
        self.size = num_neurons

    def get_size(self):
        return self.size

    def get_all_outputs(self):
        to_return = []
        for current_neuron in self.neurons:
            to_return.append(current_neuron.output)
        return to_return

    def get_expected_inner(self, num_neuron):
        aux = 0
        for current_neuron in self.neurons:
            aux += (current_neuron.weights[num_neuron] * current_neuron.delta)
        return aux

    def propagation(self, input):
        for i in range(0, self.size):
            self.neurons[i].calculate_output(input)

    def plot(self, num):
        print(str(num) + ": ")
        for i in range(0, self.size):
            self.neurons[i].plot()
        print("\n")


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
            self.layers[i].propagation(get_prev_outputs(i, input))

    def back_propagation(self, expected_value):
        for layer in range(self.num_layers - 1, 0):
            inputs = self.get_prev_outputs(layer, expected_value)
            for neuron in range(0, self.layers[layer].get_size()):
                expected = get_expected(layer, neuron, expected_value)
                self.layers[layer].neurons[neuron].update_w(inputs, expected)

    def train(self, training, expected_output, max_gen):
        current_gen = 0
        error = 0
        self.error_min = np.inf

        positions = np.arange(0, len(self.training))

        while self.error_min > self.ERROR_MIN and current_gen < max_gen:
            np.random.shuffle(positions)
            for i in positions:
                forward_propagation(training[i])
                back_propagation(expected_output[i])

            error_min = calculate_error()

    def test(self, test_set):
        pass
