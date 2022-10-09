import numpy as np

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