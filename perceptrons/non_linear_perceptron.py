from perceptrons.perceptron import Perceptron
import numpy as np

class NonLinearPerceptron(Perceptron):

    def __init__(self, training, expected_output, learning_rate):
        super().__init__(training, expected_output, learning_rate)
        self.max_value = max(expected_output)
        self.min_value = min(expected_output)

    def activation(self, excitation):
        return np.tanh(excitation)

    def error(self, w):
        error = 0
        for i in range(len(self.training)):
            output = np.inner(self.training[i], w)
            error += (self.scale(self.expected_output[i]) - self.scale(self.activation(output))) ** 2
        return error / 2

    def scale(self, value):
        return (((value + 1) / 2) * (self.max_value - self.min_value)) + self.min_value

    # TODO: Implementar la derivada de la funcion de activacion
    # def activation_derivative(self, excitation):