from numpy import exp, array, random, dot
from PIL import Image
import numpy as np


# зададим примеры


class NeuralNetwork():
    def __init__(self):
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((16, 1)) - 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)
            error = training_set_outputs - output

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


img = Image.open('o.png')
arr = np.array(img)
flat_arr = arr.ravel()
for i in range(0, len(flat_arr)):
    if flat_arr[i] == 255:
        flat_arr[i] = 1


def remove_every_other(my_list):
    return my_list[::3]


print(remove_every_other(flat_arr))
if __name__ == "__main__":
    # Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 0, 0,
                                  0, 1, 1, 0,
                                  0, 1, 1, 0,
                                  0, 0, 0, 0],
                                 [0, 1, 1, 0,
                                  1, 0, 0, 1,
                                  1, 0, 0, 1,
                                  0, 1, 1, 0],
                                 [0, 1, 1, 0,
                                  1, 0, 1, 1,
                                  1, 1, 0, 1,
                                  0, 1, 1, 0],
                                 [0, 0, 0, 0,
                                  0, 1, 1, 0,
                                  0, 1, 0, 0,
                                  0, 0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print("Considering new situation  ?: ")
    print(neural_network.think(flat_arr))
