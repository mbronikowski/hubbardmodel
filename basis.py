import numpy as np
from scipy import special


def assign_number_to_electron_number(number_of_bits):
    number_of_possible_vectors = 2 ** number_of_bits
    result = np.empty(number_of_possible_vectors, dtype=int)  # using int, max 7x7 array
    for i in range(number_of_possible_vectors):
        result[i] = bin(i).count("1")
    return result


def get_spinless_basis(number_of_electrons, number_of_holes, electron_number_array):
    it_number_array = np.nditer(electron_number_array, flags=['f_index'])
    result_len = special.comb(number_of_holes, number_of_electrons, exact=True)
    result = np.empty(result_len, dtype=int)
    it_result = np.nditer(result, flags=['f_index'])
    while not it_number_array.finished:
        if it_number_array[0] == number_of_electrons:
            result[it_result.index] = it_number_array.index
            it_result.iternext()
        it_number_array.iternext()
    return result


def get_spin_basis(number_of_electrons, number_of_positive_spins, number_of_holes, electron_number_array):
    number_of_negative_spins = number_of_electrons - number_of_positive_spins

    positive_spin_array = get_spinless_basis(number_of_positive_spins, number_of_holes, electron_number_array)
    negative_spin_array = get_spinless_basis(number_of_negative_spins, number_of_holes, electron_number_array)

    result_len = special.comb(number_of_holes, number_of_electrons, exact=True) \
               * special.comb(number_of_electrons, number_of_positive_spins, exact=True)
    result = np.empty(shape=(result_len, 2), dtype=int)

    it_result = np.nditer(result, flags=['c_index'])
    it_positive = np.nditer(positive_spin_array, flags=['f_index'])
    while not it_positive.finished:
        it_negative = np.nditer(negative_spin_array, flags=['f_index'])

        while not it_negative.finished:
            if not (it_positive[0] & it_negative[0]):
                result[it_result.index][0] = it_positive[0]
                result[it_result.index][1] = it_negative[0]
                it_result.iternext()
            it_negative.iternext()
        it_positive.iternext()
    return result
    # this mimics the implementation by Mathematica subteam
