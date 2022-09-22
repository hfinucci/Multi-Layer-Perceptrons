import numpy as np
import json
from perceptrons.step_perceptron import StepPerceptron
from Ejer1.constants import *

with open("Ejer1/ex1_config.json") as file:
    jsonObject = json.load(file)
    file.close()

learning_rate = float(jsonObject["learning_rate"])
generation = int(jsonObject["generation"])
operation = str(jsonObject["operation"])

training_set = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])

if operation == AND:
    expected_output = np.array([-1, -1, -1, 1])
else:
    expected_output = np.array([1, 1, -1, -1])


perceptron = StepPerceptron(training_set, expected_output, learning_rate)
perceptron.train(generation)
# perceptron.plot()
# print("------------------")

results = perceptron.test(np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]]))
print(results)