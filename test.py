from numpy import exp, array, random, dot
from PIL import Image
import numpy as np

training_img_1 = Image.open('learning_zero_1.png')
training_img_2 = Image.open('learning_zero_2.png')
training_img_3 = Image.open('learning_x_1.png')
training_img_4 = Image.open('learning_x_2.png')
arr1 = np.array(training_img_1)
arr2 = np.array(training_img_2)
arr3 = np.array(training_img_3)
arr4 = np.array(training_img_4)
flat_arr1 = arr1.ravel()
flat_arr2 = arr2.ravel()
flat_arr3 = arr3.ravel()
flat_arr4 = arr4.ravel()


def remove_every_other(my_list):
    for i in range(0, len(my_list)):
        if my_list[i] == 255:
            my_list[i] = 1
    return my_list[::3]
training_set_inputs = array([remove_every_other(flat_arr1),
                             remove_every_other(flat_arr2),
                             remove_every_other(flat_arr3),
                             remove_every_other(flat_arr4)])
training_set_outputs = array([[0, 0, 1, 1]]).T
random.seed(1)
synaptic_weights = 2 * random.random((100, 1)) - 1
for iteration in range(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))

img = Image.open('test zero.png')
arr = np.array(img)
flat_arr = arr.ravel()
new_arr_2 = remove_every_other(flat_arr)
for i in range(0, len(flat_arr)):
    if flat_arr[i] == 255:
        flat_arr[i] = 1
a = (1 / (1 + exp(-(dot(new_arr_2, synaptic_weights)))))
print('Вероятность того что это Х:')
print('%.3f' % a)
img.show()
