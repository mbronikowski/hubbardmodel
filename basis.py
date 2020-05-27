import numpy as np
from scipy import special, sparse


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


def get_electron_number_array():
    """Public function which shows the private LUT generated by _assign_number_to_electron_number."""
    return _electron_number_array


def get_spinless_basis(number_of_electrons, number_of_sites):
    """Generates np array of number form vectors for a given number of holes.

    The function uses the lookup table _electron_number_array. In case the array has not yet been generated,
    the function will call _assign_number_to_electron_number to generate it first.
    """
    if number_of_sites != _electron_number_array_size:
        _assign_number_to_electron_number(number_of_sites)
    it_number_array = np.nditer(_electron_number_array, flags=['f_index'])
    result_len = special.comb(number_of_sites, number_of_electrons, exact=True)
    result = np.empty(result_len, dtype=int)
    it_result = np.nditer(result, flags=['f_index'])
    while not it_number_array.finished:
        if it_number_array[0] == number_of_electrons:
            result[it_result.index] = it_number_array.index
            it_result.iternext()
        it_number_array.iternext()
    return result
    # The output array is sorted, allowing for binary searches.


def get_free_basis(number_of_electrons, number_of_positive_spins, number_of_sites):
    """Generates np array of pairs of numbers which represent positive and negative spins in a free model."""
    number_of_negative_spins = number_of_electrons - number_of_positive_spins

    positive_spin_array = get_spinless_basis(number_of_positive_spins, number_of_sites)
    if number_of_positive_spins == number_of_negative_spins:
        negative_spin_array = positive_spin_array
    else:
        negative_spin_array = get_spinless_basis(number_of_negative_spins, number_of_sites)

    result_len = positive_spin_array.size * negative_spin_array.size
    result = np.empty(shape=(result_len, 2), dtype=int)

    it_result = np.nditer(result, flags=['c_index'])                 # Iterator is defined over the entire list,
    it_positive = np.nditer(positive_spin_array, flags=['f_index'])  # but the function quits before it goes too far.
    while not it_positive.finished:                                  # TODO: fix it here and in constrained basis gen.
        it_negative = np.nditer(negative_spin_array, flags=['f_index'])
        while not it_negative.finished:
            result[it_result.index][0] = it_positive[0]
            result[it_result.index][1] = it_negative[0]
            it_result.iternext()
            it_negative.iternext()
        it_positive.iternext()
    return result
    # Vectors are sorted by increasing first, then second number.


def get_constrained_basis(number_of_electrons, number_of_positive_spins, number_of_sites):
    """Generates np array of pairs of numbers which represent positive and negative spins in a constrained model."""
    number_of_negative_spins = number_of_electrons - number_of_positive_spins

    positive_spin_array = get_spinless_basis(number_of_positive_spins, number_of_sites)
    if number_of_positive_spins == number_of_negative_spins:
        negative_spin_array = positive_spin_array
    else:
        negative_spin_array = get_spinless_basis(number_of_negative_spins, number_of_sites)

    result_len = special.comb(number_of_sites, number_of_electrons, exact=True) \
               * special.comb(number_of_electrons, number_of_positive_spins, exact=True)
    result = np.empty(shape=(result_len, 2), dtype=int)

    it_result = np.nditer(result, flags=['c_index'])                 # Iterator is defined over the entire list,
    it_positive = np.nditer(positive_spin_array, flags=['f_index'])  # but the function quits before it goes too far.
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


def get_spinless_vector_index(basis, vector):
    """Wrapper for np.searchsorted which finds the index of a spinless vector."""
    return np.searchsorted(basis, vector)


def get_spin_vector_index(basis, vector):
    """Finds the index of a spin vector in the given basis, assuming the basis is sorted.

    The function works on both the free and constrained model bases.
    The parameter "basis" needs to be a numpy array with vectors sorted first by the first (positive in this project)
    spin, then by the second (negative) spin. Such bases are generated by get_free_basis and get_constrained_basis.
    """
    lower_bound = np.searchsorted(basis[:, 0], vector[0], side='left')
    upper_bound = np.searchsorted(basis[:, 0], vector[0], side='right')
    inner_index = np.searchsorted(basis[lower_bound:upper_bound, 1], vector[1], side='left')
    return lower_bound + inner_index


_electron_hop_square_lookup = np.empty(0)
_electron_hop_square_lookup_side = 0


def _generate_square_electron_hop_lookup(side_size):
    """Generates a private array of possible hops of an electron on an empty square with side side_size."""
    global _electron_hop_square_lookup
    global _electron_hop_square_lookup_side
    if _electron_hop_square_lookup_side != side_size:
        number_of_sites = side_size ** 2
        _electron_hop_square_lookup_side = side_size
        _electron_hop_square_lookup = np.zeros(number_of_sites, dtype=int)
        for i in range(number_of_sites):
            pattern = (1 << ((i - side_size) % number_of_sites)) \
                      | (1 << ((i + side_size) % number_of_sites)) \
                      | (1 << ((i - 1) % side_size + (i // side_size) * side_size)) \
                      | (1 << ((i + 1) % side_size + (i // side_size) * side_size))
            _electron_hop_square_lookup[i] = pattern


def get_square_electron_hop_lookup():
    """Public function to view the hop lookup array."""
    return _electron_hop_square_lookup


def list_possible_spinless_square_hops(vector, number_of_electrons, side_size):
    _generate_square_electron_hop_lookup(side_size)
    number_of_sites = side_size ** 2
    resulting_vectors = np.empty((2, 4 * number_of_electrons + 1), dtype=int)
    resulting_vectors[0][0] = 0                                       # The 0th element holds number of found hops.
    for source_bit in range(number_of_sites):
        if vector >> source_bit & 1:
            hops = _electron_hop_square_lookup[source_bit] & ~ vector
            for hop_bit in range(number_of_sites):
                if hops >> hop_bit & 1:
                    resulting_vectors[0][0] += 1
                    resulting_vectors[0][resulting_vectors[0][0]] = (vector | (1 << hop_bit)) & ~ (2 ** source_bit)
                    if (source_bit - hop_bit) % 2:
                        resulting_vectors[1][resulting_vectors[0][0]] = 1
                    else:
                        resulting_vectors[1][resulting_vectors[0][0]] = -1
    result_view = resulting_vectors[:, 1 : resulting_vectors[0][0] + 1]
    return result_view[:, result_view[0, :].argsort()]


def spinless_square_hamiltonian(basis, number_of_electrons, side_size):
    hamiltonian = sparse.dok_matrix((basis.size, basis.size), dtype=np.double)
    it_basis = np.nditer(basis, flags=['f_index'])
    while not it_basis.finished:
        vecs_to_add = list_possible_spinless_square_hops(basis[it_basis.index], number_of_electrons, side_size)
        it_vecs = np.nditer(vecs_to_add[0, :], flags=['c_index'])
        while not it_vecs.finished:
            hamiltonian[it_basis.index, get_spinless_vector_index(basis, vecs_to_add[0, it_vecs.index])] \
                = vecs_to_add[1, it_vecs.index]
            it_vecs.iternext()
        it_basis.iternext()
    return hamiltonian.tocsr()


def list_possible_free_square_hops(basis, vector, number_of_positive_spins, number_of_negative_spins, side_size):
    positive_hops = list_possible_spinless_square_hops(vector[0], number_of_positive_spins, side_size)
    negative_hops = list_possible_spinless_square_hops(vector[1], number_of_negative_spins, side_size)
    resulting_vectors = np.zeros((2, positive_hops[0].size + negative_hops[0].size), dtype=int)
    negative_insert_index = np.searchsorted(positive_hops[0, :], vector[0])
    it_positive = np.nditer(positive_hops[0, :], flags=['c_index'])
    it_negative = np.nditer(negative_hops[0, :], flags=['c_index'])
    it_result = np.nditer(resulting_vectors[0, :], flags=['c_index'])
    while not it_result.finished:
        if it_result.index != negative_insert_index:
            resulting_vectors[0, it_result.index] = get_spin_vector_index(basis, [it_positive[0], vector[1]])
            resulting_vectors[1, it_result.index] = positive_hops[1, it_positive.index]
            it_positive.iternext()
            it_result.iternext()
        else:
            while not it_negative.finished:
                resulting_vectors[0, it_result.index] \
                    = get_spin_vector_index(basis, [vector[0], it_negative[0]])
                resulting_vectors[1, it_result.index] = negative_hops[1, it_negative.index]
                it_negative.iternext()
                it_result.iternext()
    return resulting_vectors


def free_square_hamiltonian(basis, number_of_electrons, number_of_positive_spins, side_size):
    number_of_negative_spins = number_of_electrons - number_of_positive_spins
    hamiltonian = sparse.dok_matrix((basis[:, 0].size, basis[:, 0].size), dtype=np.double)
    it_basis = np.nditer(basis[:, 0], flags=['f_index'])
    while not it_basis.finished:
        vecs_to_add = list_possible_free_square_hops(basis, basis[it_basis.index],
                                                     number_of_positive_spins, number_of_negative_spins, side_size)
        it_vecs = np.nditer(vecs_to_add[0, :], flags=['c_index'])
        while not it_vecs.finished:
            hamiltonian[it_basis.index, it_vecs[0]] = vecs_to_add[1, it_vecs.index]
            it_vecs.iternext()
        it_basis.iternext()
    return hamiltonian.tocsr()


def list_possible_constrained_square_hops():
    pass


def constrained_square_hamiltonian():
    pass
