# Import libraries
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import json
import numpy as np
import random

from perceptrons.perceptron_types import LINEAR, NON_LINEAR
from perceptrons.linear_perceptron import LinearPerceptron
from perceptrons.non_linear_perceptron import NonLinearPerceptron
from utils import *

x = []
y = []
z = []


with open('Ejer2/TP2-ej2-conjunto.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    data = []
    expected_output = []

    for row in plots:
        print(row)
        x.append(float(row[0]))
        y.append(float(row[2]))
        z.append(float(row[3]))
        data.append([float(row[0]), float(row[1]), float(row[2])])
        expected_output.append(float(row[3]))

    csvfile.close()

    #pnt promedio
    x1 = sum(x)/len(x)
    y1 = sum(y) / len(y)
    z1 = sum(z) / len(z)

    colors = []
    for i in x:
        colors.append("green")
    colors.append('red')

    x.append(x1)
    y.append(y1)
    z.append(z1)

    # Creating figure
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(x, y, z, color=colors)
    plt.title("simple 3D scatter plot")

    # show plot
    plt.show()

with open('Ejer2/ex2_config.json') as file:
    jsonObject = json.load(file)
    file.close()

perceptron = str(jsonObject["perceptron"])
learning_rate = float(jsonObject["learning_rate"])
beta = float(jsonObject["beta"])
generation = int(jsonObject["generation"])
do_test = bool(jsonObject["test"])

if (perceptron == LINEAR):
    perceptron = LinearPerceptron(data, expected_output, learning_rate)

elif (perceptron == NON_LINEAR):
    expected_output = escalate(expected_output)
    perceptron = NonLinearPerceptron(data, expected_output, learning_rate, beta)
    
else: 
    print("Perceptron not found")
    exit(1)

accuracies, errors, min_w, weights = perceptron.train(generation)

if (perceptron == NON_LINEAR):
    
    plot_accuracies(accuracies)

if (do_test and perceptron == NON_LINEAR):
    np.random.shuffle(positions)
    random_index = random.randint(10, 20)
    expected_input_test_set = data[random_index:27]
    print(expected_input_test_set)
    results = perceptron.test(expected_input_test_set)
    print(results)
    plot_error_in_accuracy(results, expected_output[random_index:27])
    print("Mean error: ", np.mean(abs(results - expected_output[random_index:27])))

print("Error min: " + str(min(errors)))
print("Min w: " + str(min_w))
plot_errors(errors)
