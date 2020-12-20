from numpy import exp, array, random, dot
from PIL import Image
import numpy as np

training_set_inputs = array([[0, 0, 0, 0, 0, 0,
                              0, 1, 1, 1, 1, 0,
                              0, 1, 1, 1, 1, 0,
                              0, 1, 1, 1, 1, 0,
                              0, 1, 1, 1, 1, 0,
                              0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 0,
                              1, 0, 0, 1, 0, 1,
                              1, 1, 0, 0, 0, 1,
                              1, 1, 0, 0, 1, 1,
                              1, 0, 1, 1, 0, 1,
                              0, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 0,
                              1, 0, 1, 1, 0, 1,
                              1, 1, 0, 0, 1, 1,
                              1, 1, 0, 0, 1, 1,
                              1, 0, 1, 1, 0, 1,
                              0, 1, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0,
                              0, 0, 1, 1, 0, 0,
                              0, 1, 1, 1, 1, 0,
                              0, 1, 1, 1, 1, 0,
                              0, 0, 1, 1, 0, 0,
                              0, 0, 0, 0, 0, 0]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((36, 1)) - 1
for iteration in range(100):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))

img = Image.open('o.png')
arr = np.array(img)
flat_arr = arr.ravel()
def remove_every_other(my_list):
    return my_list[::4]
new_arr = remove_every_other(flat_arr)
for i in range(0, len(new_arr)):
    if new_arr[i] == 255:
        new_arr[i] = 1
print(new_arr)
for i in range(0, len(flat_arr)):
    if flat_arr[i] == 255:
        flat_arr[i] = 1
print (1 / (1 + exp(-(dot(new_arr, synaptic_weights)))))
