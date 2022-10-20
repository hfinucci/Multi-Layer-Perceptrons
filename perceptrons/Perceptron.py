import numpy as np
from abc import ABC, abstractmethod
import random
from perceptrons.perceptron_types import NON_LINEAR

ERROR_MIN = 0.001


class Perceptron(ABC):
    # Every entry in the training set has an additional cell with value 1 -> UMBRAL
    # The weight vector also includes the UMBRAL
    def __init__(self, training, expected_output, learn_rate, perceptron_type):
        self.training = np.array(list(map(lambda t: np.append(t, [1]), training)), dtype=float)
        self.expected_output = expected_output
        self.learning_rate = learn_rate
        self.error_min = None
        self.w_min = None
        self.perceptron_type = perceptron_type

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

        weights = []
        errors = []
        accuracies = []

        rand_index = random.randint(10, 20)
        training_test_set = np.array(list(map(lambda t: t[0:(len(t) - 1)], self.training[0:rand_index])), dtype=float)
        expected_output_test_set = self.expected_output[0:rand_index]

        while self.error_min > ERROR_MIN and current_gen < max_generations:
            np.random.shuffle(positions)
            for i in positions:
                excitation = np.inner(self.training[i], w)
                activation = self.activation(excitation)

                w += self.learning_rate * (self.expected_output[i] - activation) * self.training[
                    i] * self.activation_derivative(excitation)

                error = self.error(w)

                if error < self.error_min:
                    self.error_min = error
                    self.w_min = w
                    weights.append(w)

            if (self.perceptron_type == NON_LINEAR):
                accuracies.append(self.get_accuracy(training_test_set, expected_output_test_set))

            errors.append(error)
            current_gen += 1
        return accuracies, errors, self.w_min, weights

    def test(self, test_set):
        real_input = np.array(list(map(lambda t: np.append(t, [1]), test_set)), dtype=float)
        results = []
        for i in range(len(test_set)):
            excitation = np.inner(real_input[i], self.w_min)
            results.append(self.activation(excitation))
        return results

    def get_accuracy(self, test_set, expected_output):
        results = self.test(test_set)
        correct = 0
        print("Results: ", results)
        print("Expected: ", expected_output)
        for i in range(len(results)):
            if (abs(results[i] - expected_output[i]) < 0.02):
                correct += 1
        return correct / len(results)

    def plot(self):
        print(self.training)
