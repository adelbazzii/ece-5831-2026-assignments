import numpy as np


class LogicGate:
    # initialize the logic gate opject with the specified gate type, weights, and bias
    def __init__(self, gate_type, weights, bias):
        # gate type of the logic gate (AND, OR, NAND, NOR, XOR)
        self.gate_type = gate_type
        # array containing weights for each input
        self.weights = np.array(weights)
        # bias value for the gate
        self.bias = np.array([bias])

    # compute the output of the gate
    def compute(self, inputs):
        x = np.array(inputs)
        # compute the weighted sum of inputs and bias
        a = np.dot(x, self.weights) + self.bias
        return 1 if a >= 0 else 0
