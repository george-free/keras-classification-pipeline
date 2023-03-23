import numpy as np

def get_initial_weights(output_size, degree_of_freedom=6):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, degree_of_freedom), dtype='float32')
    weights = [W, b.flatten()]
    return weights