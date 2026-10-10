import matplotlib.pyplot as plt
import numpy as np
from multilayer_perceptron import MultilayerPerceptron

# example input vector
input = np.array([5, 2, 1])

# initialize the neural network
neural_network = MultilayerPerceptron()

# perform forward propagation and retain each layer's activation values
layer_values = neural_network.forward(input, return_activations=True)
output = layer_values["Output layer"]["a"]

# print the output of the neural network
print("Output of the neural network:", output)

# weights and biases are randomly initialized, so the output will vary each time the code is ran

# plot z values and activations from every layer
figure, axes = plt.subplots(
    len(layer_values),
    2,
    figsize=(12, 10),
    squeeze=False,
)

for row, (layer_name, values) in enumerate(layer_values.items()):
    neuron_indices = range(len(values["z"]))

    axes[row, 0].bar(neuron_indices, values["z"])
    axes[row, 0].set_ylabel(layer_name)
    axes[row, 0].set_title("z (pre-activation)")

    axes[row, 1].bar(neuron_indices, values["a"])
    axes[row, 1].set_title("a (post-activation)")

    for column in range(2):
        axes[row, column].set_xlabel("Neuron")

figure.suptitle("Pre-activations and Activations by Layer")
figure.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
