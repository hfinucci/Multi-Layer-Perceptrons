import time
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

training_set = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
training_set = np.array(list(map(lambda t: np.append(t, [1]), training_set)), dtype=float)

print(training_set)

# print("_----------------------------------------")
# w = np.random.rand(len(training_set[0]))
# print(w)
# positions = np.arange(0, len(training_set))
# print(positions)
# print("_----------------------------------------")
#
#
# excitation = np.inner(training_set[0], w)
# print(excitation)