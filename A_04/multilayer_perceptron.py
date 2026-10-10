import numpy as np


class MultilayerPerceptron:
    def __init__(
        self,
        input_size=3,
        hidden1_size=5,
        hidden2_size=5,
        hidden3_size=2,
        output_size=2,
    ):
        # initialize dimensions of the layers
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.hidden3_size = hidden3_size
        self.output_size = output_size

        # initialize weights and biases for each layer
        self.weights1 = np.random.randn(self.input_size, self.hidden1_size)
        self.biases1 = np.random.randn(self.hidden1_size)
        self.weights2 = np.random.randn(self.hidden1_size, self.hidden2_size)
        self.biases2 = np.random.randn(self.hidden2_size)
        self.weights3 = np.random.randn(self.hidden2_size, self.hidden3_size)
        self.biases3 = np.random.randn(self.hidden3_size)
        self.weights4 = np.random.randn(self.hidden3_size, self.output_size)
        self.biases4 = np.random.randn(self.output_size)

    def step(self, z):
        return np.where(z >= 0, 1, 0)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def forward(self, x, return_activations=False):
        # perform forward propagation through the network
        z1 = np.dot(x, self.weights1) + self.biases1
        a1 = self.sigmoid(z1)

        z2 = np.dot(a1, self.weights2) + self.biases2
        a2 = self.sigmoid(z2)

        z3 = np.dot(a2, self.weights3) + self.biases3
        a3 = self.sigmoid(z3)

        z4 = np.dot(a3, self.weights4) + self.biases4
        a4 = self.relu(z4)

        if return_activations:
            return {
                "Layer 1": {"z": z1, "a": a1},
                "Layer 2": {"z": z2, "a": a2},
                "Layer 3": {"z": z3, "a": a3},
                "Output layer": {"z": z4, "a": a4},
            }

        return a4
