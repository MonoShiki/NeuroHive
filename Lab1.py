from pprint import pprint

import numpy as np

weighs = (np.array(
    [[w1, w2] for w1 in np.arange(start=-1, stop=1.00001, step=0.05) for w2 in np.arange(
        start=-1, stop=1.00001, step=0.05)]))

training_data = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 0),
    (np.array([1, 0]), 0),
    (np.array([1, 1]), 1),
]


def not_linear_f(S):
    return 1 if S >= 1 else 0


def perceptron(x, w):
    S = np.dot(x, w)
    Y = not_linear_f(S)
    return Y


w_ = []
for w in weighs:
    se = 0
    for x, y in training_data:
        Y = perceptron(x, w)
        error = (Y - y)
        se += error ** 2
    if se == 0:
        w_.append(w)
pprint(w_)
