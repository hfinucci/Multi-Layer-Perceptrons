from multiperceptron.multi_perceptron import MultiPerceptron
from multiperceptron.activation import *
import numpy as np
from ast import And
import json
from Ejer3.constants import *

with open("Ejer3/ex3.1_config.json") as file:
    jsonObject = json.load(file)
    file.close()

learning_rate = float(jsonObject["learning_rate"])
generation = int(jsonObject["generation"])
operation = str(jsonObject["operation"])

training_set = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])

if operation == AND:
    expected_output = np.array([[-1], [-1], [-1], [1]])
else:
    expected_output = np.array([[1], [1], [-1], [-1]])

activation = Tanh()
multi_layer = MultiPerceptron([2, 2, 1], learning_rate, activation)
error = multi_layer.train(training_set, expected_output, generation, 0.001, 1)
print('--------------------\n')
# print(multi_layer)
print('--------------------\n')
print(multi_layer.test(training_set))

# plot_graph(training_set, expected_output, min_w)
# plot_errors(errors)

# results = perceptron.test(np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]]))
# print(results)
# print("Min weight: ", min_w)
# multi_layer.save('Ejer3/weights.txt')
