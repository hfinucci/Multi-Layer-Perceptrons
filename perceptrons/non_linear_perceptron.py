from perceptrons.perceptron import Perceptron
from perceptrons.perceptron_types import NON_LINEAR
import numpy as np

class NonLinearPerceptron(Perceptron):

    def __init__(self, training, expected_output, learning_rate, beta):
        super().__init__(training, expected_output, learning_rate, NON_LINEAR)
        self.max_value = max(expected_output)
        self.min_value = min(expected_output)
        self.beta = beta

    def activation(self, excitation):
        return np.tanh(self.beta * excitation)

    def error(self, w):
        error = 0
        for i in range(len(self.training)):
            output = np.inner(self.training[i], w)
            error += (self.expected_output[i] - self.activation(output)) ** 2
        return error / 2

    def scale(self, value):
        return (((value + 1) / 2) * (self.max_value - self.min_value)) + self.min_value

    # TODO: Chequear
    def activation_derivative(self, excitation):
        return self.beta * (1 - np.tanh(self.beta * excitation) ** 2)