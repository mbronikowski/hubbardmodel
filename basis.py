import numpy as np
from scipy import special


def _assign_number_to_electron_number(number_of_bits):
    """Generates np array LUT which assigns number to its number of electrons."""
    global _electron_number_array
    global _electron_number_array_size
    number_of_possible_vectors = 2 ** number_of_bits
    _electron_number_array = np.empty(number_of_possible_vectors, dtype=int)  # using int, max 7x7 array
    for i in range(number_of_possible_vectors):
        _electron_number_array[i] = bin(i).count("1")
    _electron_number_array_size = number_of_bits


_electron_number_array_size = 0
_electron_number_array = np.empty(0)


def show_electron_number_array():
    return _electron_number_array


def get_spinless_basis(number_of_electrons, number_of_holes):
    """Generates np array of number form vectors for a given number of holes."""

    if number_of_holes != _electron_number_array_size:
        _assign_number_to_electron_number(number_of_holes)
    it_number_array = np.nditer(_electron_number_array, flags=['f_index'])
    result_len = special.comb(number_of_holes, number_of_electrons, exact=True)
    result = np.empty(result_len, dtype=int)
    it_result = np.nditer(result, flags=['f_index'])
    while not it_number_array.finished:
        if it_number_array[0] == number_of_electrons:
            result[it_result.index] = it_number_array.index
            it_result.iternext()
        it_number_array.iternext()
    return result
    # The output array is sorted, allowing for binary searches.


def get_spin_basis(number_of_electrons, number_of_positive_spins, number_of_holes):
    """Generates np array of pairs of numbers which represent positive and negative spins."""
    number_of_negative_spins = number_of_electrons - number_of_positive_spins

    positive_spin_array = get_spinless_basis(number_of_positive_spins, number_of_holes)
    if number_of_positive_spins == number_of_negative_spins:
        negative_spin_array = positive_spin_array
    else:
        negative_spin_array = get_spinless_basis(number_of_negative_spins, number_of_holes)

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
    # this mimics the implementation by the Mathematica subteam.
    # Vectors are sorted by increasing first, then second number.
