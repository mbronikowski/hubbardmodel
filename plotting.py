import numpy as np
import matplotlib as mpl


def print_square_spinless_vector(vector, side_size):
    vec_array = np.empty(shape=(side_size, side_size), dtype=np.byte)
    for i in range(side_size):
        for j in range(side_size):
            vec_array[i][j] = (vector >> (i * side_size + j)) & 1
        print(vec_array[i])


def print_square_spin_vector(vector, side_size):
    for i in range(side_size):
        print('[', end='')
        for j in range(side_size):
            if (j + 1) % side_size:
                end_char = ' '
            else:
                end_char = ''
            pos = vector[0] >> (i * side_size + j) & 1
            neg = vector[1] >> (i * side_size + j) & 1
            if pos & neg:
                print('2', end=end_char)
            elif pos:
                print('+', end=end_char)
            elif neg:
                print('-', end=end_char)
            else:
                print('0', end=end_char)
        print(']')


