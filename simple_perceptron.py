from Perceptron import Perceptron
import numpy as np


class SimplePerceptron(Perceptron):

    def __init__(self, training, expected_output, learning_rate):
        super().__init__(training, expected_output, learning_rate)

    def activation(self, excitation):
        return 1.0 if excitation >= 0.0 else -1.0

    def error(self, w):
        error = 0
        for i in range(len(self.training)):
            excitation = np.inner(self.training[i], w)
            output = self.activation(excitation)
            error += (self.expected_output[i] - output) ** 2
        return error / len(self.training)
