import numpy as np
import json
from perceptrons.step_perceptron import StepPerceptron
from Ejer1.constants import *
import matplotlib.pyplot as plt
import numpy as np

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
errors, min_w = perceptron.train(generation)
# perceptron.plot()
# print("------------------")

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('center')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# v_gradient = (min_w[1]) / (min_w[0])
# line_gradient = -1 / v_gradient

# x = np.linspace(-5,5,100)
# y = line_gradient * x + 1

# plt.plot(x, y, 'r')

# plt.plot(errors)
print(min_w)
plt.quiver(0.5, 0.5, min_w[0]+0.5, min_w[1]+0.5,color='b', units='xy', scale=100)

plt.show()
results = perceptron.test(np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]]))
print(results)
