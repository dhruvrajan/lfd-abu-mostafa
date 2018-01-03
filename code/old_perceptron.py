__author__ = 'dhruv'
"""
Perceptron Algorithm Implementation. (Also used for Homework #1

1. Assume Linearly-Separable Data.
2. Pick a random separating line
3. For each misclassified point, update the line so that it is correct.
"""

import numpy as np
import matplotlib.pyplot as mpl


def sign(x):
    return -1 if x < 0 else 1


# Target Function: y = 3 * x + 4
def target(feature_vector, type=False):
    if type:
        return .5 * feature_vector + 4
    return sign(feature_vector[1] - (.5 * feature_vector[0] + 4))


def slope(function):
    return (function(1, type=True) - function(0, type=True)) / 1


# Uses (y - y1) = m * (x-x1)
# y = ((y0 - y1)/(x0 - x1)) * (x - x1) + y1
def generate_target():
    def f(feature_vector, type=False):
        x0, x1, y0, y1 = (0, np.random.random(), 0, np.random.random())
        print(x1, y1)
        if type:
            return ((y0 - y1) / (x0 - x1)) * (feature_vector - x1) + y1
        return sign(feature_vector[1] - ((y0 - y1) / (x0 - x1)) * (feature_vector[0] - x1) - y1)

    return f


def create_linearly_separable(target_function, n_samples=100, n_features=2):
    dataset = np.zeros((n_samples, n_features))
    classification = np.zeros(n_samples)

    for x in range(n_samples):
        i = np.random.random()
        j = np.random.random()
        dataset[x][0] = i * 100
        dataset[x][1] = j * 100
        classification[x] = target_function(dataset[x])

    dataset = np.column_stack((np.array([1 for x in range(len(dataset))]), dataset))
    return dataset, classification


target_function = generate_target()
print(slope(target_function))
dataset, classification = create_linearly_separable(target_function=target_function)
weights = np.zeros(len(dataset[0]))
weights[0] = 0

X = [dataset[x][1] for x in range(len(dataset))]
y = [dataset[x][2] for x in range(len(dataset))]


def h(feature_vector, scalar=False):
    return np.inner(np.transpose(weights), feature_vector)


num_misclassified = 0
for x in range(len(dataset)):
    if not sign(h(dataset[x])) == classification[x]:
        num_misclassified += 1
        weights = np.add(weights, classification[x] * dataset[x])

# for x in range(len(dataset)):
# if not sign(h(dataset[x])) == classification[x]:
#         num_misclassified += 1
#         weights = np.add(weights, classification[x] * dataset[x])

hypothesis_classification = []
for x in range(len(dataset)):
    hypothesis_classification.append(sign(h(dataset[x])))

X0 = [dataset[x][1] for x in range(len(dataset)) if hypothesis_classification[x] == -1]
y0 = [dataset[x][2] for x in range(len(dataset)) if hypothesis_classification[x] == -1]

X1 = [dataset[x][1] for x in range(len(dataset)) if hypothesis_classification[x] == 1]
y1 = [dataset[x][2] for x in range(len(dataset)) if hypothesis_classification[x] == 1]

X0r = [dataset[x][1] for x in range(len(dataset)) if classification[x] == -1]
y0r = [dataset[x][2] for x in range(len(dataset)) if classification[x] == -1]

X1r = [dataset[x][1] for x in range(len(dataset)) if classification[x] == 1]
y1r = [dataset[x][2] for x in range(len(dataset)) if classification[x] == 1]


# Plot Data, Output Results
print("Misclassified: ", num_misclassified)
print(weights)
numrange = np.arange(min(X), max(X), 0.2)
# numrange2 = np.column_stack((numrange, np.zeros(len(numrange))))
# mpl.plot(numrange, h(numrange))
mpl.plot(numrange, target_function(numrange, type=True))

mpl.scatter(X0, y0, color="blue")
mpl.scatter(X1, y1, color="red")
print("showing")
mpl.show()
mpl.figure(1)
mpl.plot(numrange, target_function(numrange, type=True))

mpl.scatter(X0r, y0r, color="blue")
mpl.scatter(X1r, y1r, color="red")
mpl.show()
