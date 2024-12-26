import numpy as np


def step_function(x):
    return np.where(x >= 0, 1, 0)


X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Y = np.array([[0],
              [1],
              [1],
              [1]])


np.random.seed(0)
weights = np.random.rand(X.shape[1], 1)
bias = np.random.rand(1)


learning_rate = 0.1
epochs = 10


for epoch in range(epochs):
    for i in range(X.shape[0]):

        linear_output = np.dot(X[i], weights) + bias
        predicted = step_function(linear_output)


        error = Y[i] - predicted
        weights += learning_rate * error * X[i].reshape(-1, 1)
        bias += learning_rate * error


print("Testing the OR Gate model:")
for i in range(X.shape[0]):
    linear_output = np.dot(X[i], weights) + bias
    predicted = step_function(linear_output)
    print(f"Input: {X[i]}, Predicted Output: {predicted[0]}")