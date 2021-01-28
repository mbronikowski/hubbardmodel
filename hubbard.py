import cmath
import numpy as np
from scipy import special, sparse
from scipy.sparse import linalg


_numeric_threshold = 1e-8


class _LUTManager:
    """LUTManager is the class for LUTM, this project's lookup table manager.

    Any LUT access in the module is managed by this class. It holds the dictionaries for arrays which assign
    the number of electrons to each bit representation of a given length and for a lookup table of possible hops made
    by an electron on a square grid."""

    def __init__(self):
        self._electron_number_dict = {}  # Assigns bit representation to the number of electrons it contains.
        self._square_hop_dict = {}  # Contains bit representation of possible hops of an electron on an empty grid.

    def get_electron_number_lut(self, number_of_bits):
        if number_of_bits not in self._electron_number_dict:
            self.add_electron_number_lut(number_of_bits)
        return self._electron_number_dict[number_of_bits]

    def get_square_hop_lut(self, side_size):
        if side_size not in self._square_hop_dict:
            self.add_square_hop_lut(side_size)
        return self._square_hop_dict[side_size]

    def add_electron_number_lut(self, number_of_bits):
        number_of_possible_vectors = 2 ** number_of_bits
        new_electron_number_array = np.empty(number_of_possible_vectors, dtype=int)   # The array uses int, allowing for
        for i in range(number_of_possible_vectors):                                   # at most a 5x5 atom grid.
            new_electron_number_array[i] = bin(i).count("1")                          # More would require 'long'.
        self._electron_number_dict[number_of_bits] = new_electron_number_array

    def add_square_hop_lut(self, side_size):
        number_of_sites = side_size ** 2
        new_square_hop_lut = np.zeros(number_of_sites, dtype=int)
        for i in range(number_of_sites):
            pattern = (1 << ((i - side_size) % number_of_sites)) \
                      | (1 << ((i + side_size) % number_of_sites)) \
                      | (1 << ((i - 1) % side_size + (i // side_size) * side_size)) \
                      | (1 << ((i + 1) % side_size + (i // side_size) * side_size))
            new_square_hop_lut[i] = pattern
        self._square_hop_dict[side_size] = new_square_hop_lut


_LUTM = _LUTManager()


class SquareSpinlessModel:
    def __init__(self, number_of_electrons, number_of_positive_spins, side_size):
        self.no_electrons = number_of_electrons
        self.no_plus_spins = number_of_positive_spins
        self.no_minus_spins = number_of_electrons - number_of_positive_spins
        self.side = side_size
        self.no_atoms = side_size ** 2
        self.basis = generate_spinless_basis(self.no_electrons, self.no_atoms)
        self.hmltn_linop = sparse.linalg.LinearOperator((self.basis.shape[0],
                                                         self.basis.shape[0]),
                                                        matvec=self._multiply_vec_no_hmltn)
        self.hamiltonian_exists = False
        self.hamiltonian = None
        self.spin_type = "spinless"

    def _multiply_vec_no_hmltn(self, vec):
        """Multiplies a vector by the model's spinless Hamiltonian without its explicit representation."""
        result = np.zeros(vec.shape[0], dtype=vec.dtype)
        it_vec = np.nditer(vec, flags=['f_index'])
        while not it_vec.finished:
            if vec[it_vec.index] != 0:
                vecs_to_add = list_possible_spinless_square_hops(self.basis[it_vec.index],
                                                                 self.no_electrons, self.side)
                for i in range(vecs_to_add.shape[1]):
                    result[get_spinless_vector_index(self.basis, vecs_to_add[0][i])] \
                        += vec[it_vec.index] * vecs_to_add[1][i]
            it_vec.iternext()
        return result

    def multiply_vec_hmltn(self, vec):
        if self.hamiltonian_exists:
            return self.hamiltonian.multiply(vec)
        return self.hmltn_linop.matvec(vec)

    def get_ground_state(self):
        if self.hamiltonian_exists:
            return sparse.linalg.eigsh(self.hamiltonian, k=1, which='SA')
        return sparse.linalg.eigsh(self.hmltn_linop, k=1, which='SA')

    def _generate_hamiltonian(self):
        self.hamiltonian = spinless_square_hamiltonian(self.basis, self.no_electrons, self.side)
        self.hamiltonian_exists = True


class SquareFreeModel:
    def __init__(self, number_of_electrons, number_of_positive_spins, side_size):
        self.no_electrons = number_of_electrons
        self.no_plus_spins = number_of_positive_spins
        self.no_minus_spins = number_of_electrons - number_of_positive_spins
        self.side = side_size
        self.no_atoms = side_size ** 2
        self.basis = generate_free_basis(self.no_electrons, self.no_plus_spins, self.no_atoms)
        self.hmltn_linop = sparse.linalg.LinearOperator((self.basis.shape[0],
                                                         self.basis.shape[0]),
                                                        matvec=self._multiply_vec_no_hmltn)
        self.hamiltonian_exists = False
        self.hamiltonian = None
        self.spin_type = "free"

    def _multiply_vec_no_hmltn(self, vec):  # TODO: TEST THIS FUNCTION
        """Multiplies a vector by the model's free Hamiltonian without its explicit representation."""
        result = np.zeros(vec.shape[0], dtype=vec.dtype)
        it_vec = np.nditer(vec, flags=['f_index'])
        while not it_vec.finished:
            if vec[it_vec.index] != 0:
                vecs_to_add = list_possible_free_square_hop_indices(self.basis, self.basis[it_vec.index],
                                                                    self.no_plus_spins, self.no_minus_spins, self.side)
                for i in range(vecs_to_add.shape[1]):
                    result[vecs_to_add[0][i]] += vec[it_vec.index] * vecs_to_add[1][i]
            it_vec.iternext()
        return result

    def multiply_vec_hmltn(self, vec):
        if self.hamiltonian_exists:
            return self.hamiltonian.multiply(vec)
        return self.hmltn_linop.matvec(vec)

    def get_ground_state(self):
        if self.hamiltonian_exists:
            return sparse.linalg.eigsh(self.hamiltonian, k=1, which='SA')
        return sparse.linalg.eigsh(self.hmltn_linop, k=1, which='SA')

    def _generate_hamiltonian(self):
        self.hamiltonian = free_square_hamiltonian(self.basis, self.no_electrons,
                                                   self.no_plus_spins, self.side)
        self.hamiltonian_exists = True


class SquareConstrainedModel:
    def __init__(self, number_of_electrons, number_of_positive_spins, side_size):
        self.no_electrons = number_of_electrons
        self.no_plus_spins = number_of_positive_spins
        self.no_minus_spins = number_of_electrons - number_of_positive_spins
        self.side = side_size
        self.no_atoms = side_size ** 2
        self.basis = generate_constrained_basis(self.no_electrons, self.no_plus_spins, self.no_atoms)
        self.hmltn_linop = sparse.linalg.LinearOperator((self.basis.shape[0],
                                                         self.basis.shape[0]),
                                                        matvec=self._multiply_vec_no_hmltn)
        self.hamiltonian_exists = False
        self.hamiltonian = None
        self.spin_type = "constrained"

    def _multiply_vec_no_hmltn(self, vec):  # TODO: TEST THIS FUNCTION
        """Multiplies a vector by the model's constrained Hamiltonian without its explicit representation."""
        result = np.zeros(vec.shape[0], dtype=vec.dtype)
        it_vec = np.nditer(vec, flags=['f_index'])
        while not it_vec.finished:
            if vec[it_vec.index] != 0:
                vecs_to_add = list_possible_constrained_square_hop_indices(self.basis,
                                                                           self.basis[it_vec.index],
                                                                           self.no_plus_spins, self.no_minus_spins,
                                                                           self.side)
                for i in range(vecs_to_add.shape[1]):
                    result[vecs_to_add[0][i]] += vec[it_vec.index] * vecs_to_add[1][i]
            it_vec.iternext()
        return result

    def multiply_vec_hmltn(self, vec):
        if self.hamiltonian_exists:
            return self.hamiltonian.multiply(vec)
        return self.hmltn_linop.matvec(vec)

    def get_ground_state(self):
        if self.hamiltonian_exists:
            return sparse.linalg.eigsh(self.hamiltonian, k=1, which='SA')
        return sparse.linalg.eigsh(self.hmltn_linop, k=1, which='SA')

    def _generate_hamiltonian(self):
        self.hamiltonian = constrained_square_hamiltonian(self.basis, self.no_electrons,
                                                          self.no_plus_spins, self.side)
        self.hamiltonian_exists = True


def _normalize(vec):
    """Returns a tuple with the norm of a vector and the normalized vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return norm, vec
    return norm, vec/norm


def generate_spinless_basis(number_of_electrons, number_of_sites):
    """Generates np array of number form vectors for a given number of holes.

    The function uses the lookup table _electron_number_array. In case the array has not yet been generated,
    the function will call _assign_number_to_electron_number to generate it first.
    """
    electron_number_array = _LUTM.get_electron_number_lut(number_of_sites)
    it_number_array = np.nditer(electron_number_array, flags=['f_index'])
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


def generate_free_basis(number_of_electrons, number_of_positive_spins, number_of_sites):
    """Generates np array of pairs of numbers which represent positive and negative spins in a free model."""
    number_of_negative_spins = number_of_electrons - number_of_positive_spins

    positive_spin_array = generate_spinless_basis(number_of_positive_spins, number_of_sites)
    if number_of_positive_spins == number_of_negative_spins:
        negative_spin_array = positive_spin_array
    else:
        negative_spin_array = generate_spinless_basis(number_of_negative_spins, number_of_sites)

    result_len = positive_spin_array.size * negative_spin_array.size
    result = np.empty(shape=(result_len, 2), dtype=int)

    it_result = np.nditer(result, flags=['c_index'])  # Iterator is defined over the entire list,
    it_positive = np.nditer(positive_spin_array, flags=['f_index'])  # but the function quits before it goes too far.
    while not it_positive.finished:  # TODO: fix it here and in constrained basis gen.
        it_negative = np.nditer(negative_spin_array, flags=['f_index'])
        while not it_negative.finished:
            result[it_result.index][0] = it_positive[0]
            result[it_result.index][1] = it_negative[0]
            it_result.iternext()
            it_negative.iternext()
        it_positive.iternext()
    return result
    # Vectors are sorted by increasing first, then second number.


def generate_constrained_basis(number_of_electrons, number_of_positive_spins, number_of_sites):
    """Generates np array of pairs of numbers which represent positive and negative spins in a constrained model."""
    number_of_negative_spins = number_of_electrons - number_of_positive_spins

    positive_spin_array = generate_spinless_basis(number_of_positive_spins, number_of_sites)
    if number_of_positive_spins == number_of_negative_spins:
        negative_spin_array = positive_spin_array
    else:
        negative_spin_array = generate_spinless_basis(number_of_negative_spins, number_of_sites)

    result_len = special.comb(number_of_sites, number_of_electrons, exact=True) \
               * special.comb(number_of_electrons, number_of_positive_spins, exact=True)
    result = np.empty(shape=(result_len, 2), dtype=int)

    it_result = np.nditer(result, flags=['c_index'])  # Iterator is defined over the entire list,
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


def list_possible_spinless_square_hops(vector, number_of_electrons, side_size):
    """Returns all possible vectors (in vector form) for any given hop along with its sign."""
    square_hop_lookup = _LUTM.get_square_hop_lut(side_size)
    number_of_sites = side_size ** 2
    resulting_vectors = np.empty((2, 4 * number_of_electrons + 1), dtype=int)
    resulting_vectors[0][0] = 0  # The 0th element holds the number of found hops.
    for source_bit in range(number_of_sites):
        if vector >> source_bit & 1:
            hops = square_hop_lookup[source_bit] & ~ vector
            for hop_bit in range(number_of_sites):
                if hops >> hop_bit & 1:
                    resulting_vectors[0][0] += 1
                    resulting_vectors[0][resulting_vectors[0][0]] = (vector | (1 << hop_bit)) & ~ (2 ** source_bit)
                    if (source_bit - hop_bit) % 2:
                        resulting_vectors[1][resulting_vectors[0][0]] = -1
                    else:
                        resulting_vectors[1][resulting_vectors[0][0]] = 1
    result_view = resulting_vectors[:, 1: resulting_vectors[0][0] + 1]
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


def list_possible_free_square_hop_indices(basis, vector, number_of_positive_spins, number_of_negative_spins, side_size):
    positive_hops = list_possible_spinless_square_hops(vector[0], number_of_positive_spins, side_size)
    negative_hops = list_possible_spinless_square_hops(vector[1], number_of_negative_spins, side_size)
    resulting_vectors = np.empty((2, positive_hops[0].size + negative_hops[0].size), dtype=int)
    negative_insert_index = np.searchsorted(positive_hops[0, :], vector[0])
    it_positive = np.nditer(positive_hops[0, :], flags=['c_index'])
    it_negative = np.nditer(negative_hops[0, :], flags=['c_index'])
    it_result = np.nditer(resulting_vectors[0, :], flags=['c_index'])
    while not it_result.finished:
        if it_result.index != negative_insert_index:
            resulting_vectors[0, it_result.index] = get_spin_vector_index(basis, (it_positive[0], vector[1]))
            resulting_vectors[1, it_result.index] = positive_hops[1, it_positive.index]
            it_positive.iternext()
            it_result.iternext()
        else:
            while not it_negative.finished:
                resulting_vectors[0, it_result.index] \
                    = get_spin_vector_index(basis, (vector[0], it_negative[0]))
                resulting_vectors[1, it_result.index] = negative_hops[1, it_negative.index]
                it_negative.iternext()
                it_result.iternext()
    return resulting_vectors


def free_square_hamiltonian(basis, number_of_electrons, number_of_positive_spins, side_size):
    number_of_negative_spins = number_of_electrons - number_of_positive_spins
    hamiltonian = sparse.dok_matrix((basis[:, 0].size, basis[:, 0].size), dtype=np.double)
    it_basis = np.nditer(basis[:, 0], flags=['f_index'])
    while not it_basis.finished:
        vecs_to_add = list_possible_free_square_hop_indices(basis, basis[it_basis.index],
                                                            number_of_positive_spins, number_of_negative_spins,
                                                            side_size)
        it_vecs = np.nditer(vecs_to_add[0, :], flags=['c_index'])
        while not it_vecs.finished:
            hamiltonian[it_basis.index, it_vecs[0]] = vecs_to_add[1, it_vecs.index]
            it_vecs.iternext()
        it_basis.iternext()
    return hamiltonian.tocsr()


def list_possible_constrained_square_hop_indices(basis, vector, number_of_positive_spins,
                                                 number_of_negative_spins, side_size):
    positive_hops = list_possible_spinless_square_hops(vector[0], number_of_positive_spins, side_size)
    negative_hops = list_possible_spinless_square_hops(vector[1], number_of_negative_spins, side_size)
    resulting_vectors = np.empty((2, positive_hops[0].size + negative_hops[0].size), dtype=int)
    negative_insert_index = np.searchsorted(positive_hops[0, :], vector[0])
    it_positive = np.nditer(positive_hops[0, :], flags=['c_index'])
    it_negative = np.nditer(negative_hops[0, :], flags=['c_index'])
    it_result = np.nditer(resulting_vectors[0, :], flags=['c_index'])
    while not (it_positive.finished and it_negative.finished):
        if (not it_positive.finished) and it_positive.index != negative_insert_index:
            if not (it_positive[0] & vector[1]):
                resulting_vectors[0, it_result.index] = get_spin_vector_index(basis, (it_positive[0], vector[1]))
                resulting_vectors[1, it_result.index] = positive_hops[1, it_positive.index]
                it_result.iternext()
            it_positive.iternext()
        else:
            while not it_negative.finished:
                if not (vector[0] & it_negative[0]):
                    resulting_vectors[0, it_result.index] \
                        = get_spin_vector_index(basis, (vector[0], it_negative[0]))
                    resulting_vectors[1, it_result.index] = negative_hops[1, it_negative.index]
                    it_result.iternext()
                it_negative.iternext()
            negative_insert_index = -1
    if it_result.finished:
        return resulting_vectors
    return resulting_vectors[:, 0: it_result.index]


def constrained_square_hamiltonian(basis, number_of_electrons, number_of_positive_spins, side_size):
    number_of_negative_spins = number_of_electrons - number_of_positive_spins
    hamiltonian = sparse.dok_matrix((basis[:, 0].size, basis[:, 0].size), dtype=np.double)
    it_basis = np.nditer(basis[:, 0], flags=['f_index'])
    while not it_basis.finished:
        vecs_to_add = list_possible_constrained_square_hop_indices(basis, basis[it_basis.index],
                                                                   number_of_positive_spins, number_of_negative_spins,
                                                                   side_size)
        it_vecs = np.nditer(vecs_to_add[0, :], flags=['c_index'])
        while not it_vecs.finished:
            hamiltonian[it_basis.index, it_vecs[0]] = vecs_to_add[1, it_vecs.index]
            it_vecs.iternext()
        it_basis.iternext()
    return hamiltonian.tocsr()


def cr_an_operator_ampl(sign, side_size, atom_index, k_list):
    """Computes the amplitude of the creation/annihilation operator in real space when
     the operator is originally in the reciprocal space.

     Sign should be 1 for creation, -1 for annihilation. Used in computing a_k^dag into a_(atom index)^dag.
     """
    return cmath.exp(sign * 2j * cmath.pi / side_size * (k_list[0] * (atom_index % side_size)
                                                         + k_list[1] * (atom_index // side_size)))


def spinless_abs_ref_state(model_abs, model_orig, k_list):
    """Generates the 0th reference state for spinless absorption, i. e. a_k^dag * ground state."""
    # abs - absorbed, orig - original
    ground_state = model_orig.get_ground_state()[1]
    result_vector = np.zeros(model_abs.basis.shape[0], dtype=complex)
    for orig_index in range(ground_state.shape[0]):
        orig_vector = model_orig.basis[orig_index]
        for atom_index in range(model_abs.no_atoms):
            if not orig_vector & (1 << atom_index):    # "If the atom is free to absorb an electron"
                abs_vector_index = get_spinless_vector_index(model_abs.basis, orig_vector + (1 << atom_index))
                result_vector[abs_vector_index] += ground_state[orig_index] \
                                                 * cr_an_operator_ampl(1, model_abs.side, atom_index, k_list)
    return _normalize(result_vector)


def spectral_green_lanczos(model, abs_em_type, ref_vec_with_norm, ground_state_energy, omega):
    """Calculates the Green function using the Lanczos method."""
    if abs_em_type == 'a':
        sign = -1
    elif abs_em_type == 'e':
        sign = 1
    else:
        raise ValueError("abs_em_type must be 'a' for absorption or 'e' for emission.")

    norm_current, ref_vec_current = ref_vec_with_norm
    ref_vec_prev = np.zeros(ref_vec_current.shape, dtype=ref_vec_current.dtype)
    norm_energy_array = np.empty((ref_vec_current.shape[0], 2))     # Holds k_i^2 and Delta eps_i

    last_i = ref_vec_current.shape[0] - 1
    for i in range(ref_vec_current.shape[0]):
        norm_energy_array[i][0] = norm_current ** 2
        ref_vec_multiplied_by_hmltn = model.multiply_vec_hmltn(ref_vec_current)
        energy_diff = (np.vdot(ref_vec_current, ref_vec_multiplied_by_hmltn) - ground_state_energy).real
        # Casting to real only deletes numerical residue.
        ref_vec_new = ref_vec_multiplied_by_hmltn \
                    - (ground_state_energy + energy_diff) * ref_vec_current \
                    - norm_current * ref_vec_prev
        ref_vec_prev = ref_vec_current
        norm_current, ref_vec_current = _normalize(ref_vec_new)

        norm_energy_array[i][1] = energy_diff

        if norm_current < _numeric_threshold:
            last_i = i
            break

    result_fraction = 0
    for i in range(last_i, -1, -1):     # Iterates from last i to zero, inclusive.
        result_fraction = norm_energy_array[i][0] / (omega + sign * norm_energy_array[i][1] - result_fraction)
    return result_fraction


# EVERYTHING BELOW IS BROKEN, TO BE FIXED


def free_spectral_absorption(basis_abs, basis_orig, ground_state, side_size, k_list):
    # abs - absorbed, orig - original
    number_of_atoms = side_size ** 2
    result_vector = np.zeros((basis_abs.shape[0], 2), dtype=complex)
    for orig_index in range(ground_state.shape[0]):
        orig_vector = basis_orig[orig_index]
        for atom_index in range(number_of_atoms):
            can_absorb_plus = ~ orig_vector[0] & (1 << atom_index)
            can_absorb_minus = ~ orig_vector[1] & (1 << atom_index)
            if can_absorb_plus:
                new_vector_index = get_spin_vector_index(basis_abs, (orig_vector[0] + can_absorb_plus, orig_vector[1]))
                result_vector[new_vector_index][0] += ground_state[orig_index] \
                                                    * cr_an_operator_ampl(1, side_size, atom_index, k_list)
            if can_absorb_minus:
                new_vector_index = get_spin_vector_index(basis_abs, (orig_vector[0], orig_vector[1] + can_absorb_minus))
                result_vector[new_vector_index][1] += ground_state[orig_index] \
                                                    * cr_an_operator_ampl(1, side_size, atom_index, k_list)
    result = 0
    for result_index in range(result_vector.shape[0]):
        result += abs(result_vector[result_index][0]) ** 2 + abs(result_vector[result_index][1]) ** 2
    return result


def constrained_spectral_absorption(basis_abs, basis_orig, ground_state, side_size, k_list):
    # abs - absorbed, orig - original
    number_of_atoms = side_size ** 2
    result_vector = np.zeros((basis_abs.shape[0], 2), dtype=complex)
    for orig_index in range(ground_state.shape[0]):
        orig_vector = basis_orig[orig_index]
        for atom_index in range(number_of_atoms):
            can_absorb = ~ (orig_vector[0] | orig_vector[1]) & (1 << atom_index)
            if can_absorb:
                new_vector_index = get_spin_vector_index(basis_abs, (orig_vector[0] + can_absorb, orig_vector[1]))
                result_vector[new_vector_index][0] += ground_state[orig_index] \
                                                    * cr_an_operator_ampl(1, side_size, atom_index, k_list)
                new_vector_index = get_spin_vector_index(basis_abs, (orig_vector[0], orig_vector[1] + can_absorb))
                result_vector[new_vector_index][1] += ground_state[orig_index] \
                                                    * cr_an_operator_ampl(1, side_size, atom_index, k_list)
    result = 0
    for result_index in range(result_vector.shape[0]):
        result += abs(result_vector[result_index][0]) ** 2 + abs(result_vector[result_index][1]) ** 2
    return result


def spinless_spectral_emission(basis_em, basis_orig, ground_state, side_size, k_list):
    # em - post-emission, orig - original
    number_of_atoms = side_size ** 2
    result_vector = np.zeros(basis_em.shape[0], dtype=complex)
    for orig_index in range(ground_state.shape[0]):
        orig_vector = basis_orig[orig_index]
        for atom_index in range(number_of_atoms):
            if orig_vector & (1 << atom_index):    # "If the atom is free to absorb an electron"
                em_vector_index = get_spinless_vector_index(basis_em, orig_vector - (1 << atom_index))
                result_vector[em_vector_index] += ground_state[orig_index] \
                                                * cr_an_operator_ampl(-1, side_size, atom_index, k_list)
    result = 0
    for result_index in range(result_vector.shape[0]):
        result += abs(result_vector[result_index]) ** 2
    return result


def spin_spectral_emission(basis_em, basis_orig, ground_state, side_size, k_list):
    # em - post-emission, orig - original
    number_of_atoms = side_size ** 2
    result_vector = np.zeros((basis_em.shape[0], 2), dtype=complex)
    for orig_index in range(ground_state.shape[0]):
        orig_vector = basis_orig[orig_index]
        for atom_index in range(number_of_atoms):
            can_emit_plus = orig_vector[0] & (1 << atom_index)
            can_emit_minus = orig_vector[1] & (1 << atom_index)
            if can_emit_plus:
                new_vector_index = get_spin_vector_index(basis_em, (orig_vector[0] - can_emit_plus, orig_vector[1]))
                result_vector[new_vector_index][0] += ground_state[orig_index] \
                                                    * cr_an_operator_ampl(-1, side_size, atom_index, k_list)
            if can_emit_minus:
                new_vector_index = get_spin_vector_index(basis_em, (orig_vector[0], orig_vector[1] - can_emit_minus))
                result_vector[new_vector_index][1] += ground_state[orig_index] \
                                                    * cr_an_operator_ampl(-1, side_size, atom_index, k_list)
    result = 0
    for result_index in range(result_vector.shape[0]):
        result += abs(result_vector[result_index][0]) ** 2 + abs(result_vector[result_index][1]) ** 2
    return result

