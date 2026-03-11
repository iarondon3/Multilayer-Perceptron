import numpy as np

def sigmoid_function(x):

    calculated_e = np.exp(-x)
    total_sum = 1 + calculated_e 
    sigmoid_value = 1 / total_sum

    return sigmoid_value


def sigmoid_derivative(layer_output):
    calculated_derivative = layer_output * (1 - layer_output)

    return calculated_derivative
