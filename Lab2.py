import numpy as np
from pprint import pprint

class Neuro:
    def __init__(self, len_x):
        np.random.seed(5)
        self.activation_function = lambda x: 1 if x >= 0 else 0
        self.weights = np.random.rand(len_x) / 10 * np.array([np.random.randint(-1, 1) for i in range(len_x)])

    def train(self, training_data, epochs, learning_rate):
        summary = []
        for _ in range(epochs):
            SE = 0
            for x, target in training_data:
                prediction = self.predict(x)
                error = target - prediction
                SE += error ** 2
                self.weights += x * error * learning_rate
            summary.append(SE)
        print(f"{summary}")

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        threshold = 0.2
        output = self.activation_function(weighted_sum - threshold)
        return output


epoch = 10

training_data = [
    (np.array([0, 0, 1]), 1),
    (np.array([1, 0, 0]), 1),
    (np.array([1, 1, 0]), 1),
    (np.array([1, 1, 1]), 1),
    (np.array([0, 0, 1]), 1),
    (np.array([0, 1, 1]), 1),
    (np.array([0, 1, 0]), 1),
    (np.array([0, 0, 0]), 0),
]

netw = Neuro(3)
netw.train(training_data=training_data, epochs=epoch, learning_rate=0.1)
print(f"Predicion : {netw.predict(np.array([0, 0, 0]))}")
