from perceptrons.Perceptron import Perceptron
import numpy as np

class NonLinearPerceptron(Perceptron):

    def __init__(self, training, expected_output, learning_rate):
        super().__init__(training, expected_output, learning_rate)

    def activation(self, excitation):
        return np.tanh(excitation)

    def error(self, w):
        error = 0
        for i in range(len(self.training)):
            output = np.inner(self.training[i], w)
            error += (self.expected_output[i] - output) ** 2
        return error / 2