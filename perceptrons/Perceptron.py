import numpy as np
from abc import ABC, abstractmethod

ERROR_MIN = 0.001

class Perceptron(ABC):

    def __init__(self, training, expected_output, learn_rate):
        self.training = np.array(list(map(lambda t: np.append(t, [1]), training)), dtype=float)
        self.expected_output = expected_output
        self.learning_rate = learn_rate
        self.error_min = None
        self.w_min = None

    @abstractmethod
    def error(self, w):
        pass

    @abstractmethod
    def activation(self, excitation):
        pass

    def activation_derivative(self, excitation):
        return 1

    def train(self, max_generations):
        current_gen = 0
        error = 0
        self.error_min = np.inf

        w = np.random.rand(len(self.training[0]))
        positions = np.arange(0, len(self.training))

        errors = []

        while self.error_min > ERROR_MIN and current_gen < max_generations:
            np.random.shuffle(positions)
            for i in positions:
                excitation = np.inner(self.training[i], w)
                activation = self.activation(excitation)
                w += self.learning_rate * (self.expected_output[i] - activation) * self.training[i] * self.activation_derivative(excitation)

                error = self.error(w)

                if error < self.error_min:
                    self.error_min = error
                    self.w_min = w
                    errors.append(float(error))

            current_gen += 1
        return errors, self.w_min

    def test(self, test_set):
        real_input = np.array(list(map(lambda t: np.append(t, [1]), test_set)), dtype=float)
        results = []
        for i in range(len(test_set)):
            excitation = np.inner(real_input[i], self.w_min)
            results.append(self.activation(excitation))
        return results

    def plot(self):
        print(self.training)

    # # Aplica la funcion de activacion y devuelve el O
    #
    # def calculate_activation(self):
    #     total = 0
    #     for i in range(1, self.ws.length):
    #         total += self.ws[i] * self.training[i]
    #     total -= self.umbral
    #     return 1 if total < 0 else -1
    #
    # def calculate_error(self, expected, activation, w, u):
    #     total = 0
    #     for i in range(0, stimulus.length):
    #         total += abs(expected - activation)
    #
    # def delta(self, expected, calc_value, stimulus):
    #     return self.learn_rate * (expected - calc_value) * stimulus
