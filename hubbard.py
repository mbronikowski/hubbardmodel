import numpy as np
from scipy import sparse
# import cmath


_numeric_threshold = 1e-8
_num_threshold_ground_state = 0.01


class _LUTManager:
    """LUTManager is the class for LUTM, this project's lookup table manager.

    Any LUT access in the module is managed by this class. It holds the dictionaries for arrays which assign
    the number of electrons to each bit representation of a given length and for a lookup table of possible hops made
    by an electron on a square grid."""

    def __init__(self):
        self._bit_number_dict = self._gen_bit_number_dict()
        self._square_hop_dict = {}  # Contains bit representation of possible hops of an electron on an empty grid.
        self._bits_between_bit_pair_dict = {}

    @property
    def bit_number_dict(self):
        return self._bit_number_dict

    def get_square_hop_lut(self, side_size):
        if side_size not in self._square_hop_dict:
            self._add_square_hop_lut(side_size)
        return self._square_hop_dict[side_size]

    def get_bits_between_bit_pair_dict(self, dtype):
        if dtype not in self._bits_between_bit_pair_dict:
            self._gen_bits_between_bit_pair_dict(dtype)
        return self._bits_between_bit_pair_dict[dtype]

    def _add_square_hop_lut(self, side):
        grid_size = side ** 2
        if grid_size < 32:
            dtype = np.uint32
        else:
            dtype = np.uint64
        new_square_hop_lut = np.zeros(grid_size, dtype=dtype)
        for i in range(grid_size):
            pattern = (1 << ((i - side) % grid_size)) \
                      | (1 << ((i + side) % grid_size)) \
                      | (1 << ((i - 1) % side + (i // side) * side)) \
                      | (1 << ((i + 1) % side + (i // side) * side))
            new_square_hop_lut[i] = pattern
        self._square_hop_dict[side] = new_square_hop_lut

    def _gen_bits_between_bit_pair_dict(self, dtype):
        bit_no = np.iinfo(dtype).bits
        result_dict = {}
        for i in range(1, bit_no):
            for j in range(i):
                key = dtype((1 << i) + (1 << j))
                bits_between = dtype(((1 << i) - 1) - ((1 << j) - 1) - (1 << j))
                result_dict[key] = bits_between
        self._bits_between_bit_pair_dict[dtype] = result_dict

    @staticmethod
    def _gen_bit_number_dict():
        number_dict = {}
        for bit_count in range(17):       # 16 is a valid number of bits
            number_dict[bit_count] = []   # Initialize the dictionary
        for number in range(2 ** 16):
            number_dict[number.bit_count()].append(number)  # Sort numbers into bins
        for i in range(17):
            number_dict[i] = np.array(number_dict[i], dtype=np.uint32)
        return number_dict


_LUTM = _LUTManager()


class _SquareModel:
    """Parent class for square models."""
    def __init__(self, side_length, electrons, positive_spins, force_dtype=None,
                 use_sparse_hmltn=None, force_hmltn_dtype=None):
        self._electrons = electrons
        self._side = side_length
        self._grid_size = side_length ** 2

        if force_dtype is not None:
            self._basis_dtype = force_dtype
        # elif self._grid_size <= 8:            # Worst case basis (Free model, 4x4 grid, 8 spin up & spin down
        #     self._basis_dtype = np.uint8      # electrons) produces small enough basis that this default makes
        # elif self._grid_size <= 16:           # no sense.
        #     self._basis_dtype = np.uint16
        elif self._grid_size <= 32:
            self._basis_dtype = np.uint32
        elif self._grid_size <= 64:
            self._basis_dtype = np.uint64
        else:
            raise ValueError("Models with shapes beyond 8x8 are not supported. Side must be 8 or lower.")

        self._positive_spins = positive_spins
        self._negative_spins = electrons - positive_spins
        if self._negative_spins < 0:
            raise ValueError(f"Number of negative spin electrons cannot be negative: {self._negative_spins}")
        self._spin_type = None
        self._basis = None
        self._inverse_basis_dict = None

        self._hamiltonian = None

        if use_sparse_hmltn is None:
            if side_length < 4:             # Hopefully sane default
                self._hamiltonian_is_sparse = False
            else:
                self._hamiltonian_is_sparse = True
        else:
            self._hamiltonian_is_sparse = use_sparse_hmltn

        if force_hmltn_dtype is None:
            self._hamiltonian_dtype = int
        else:
            self._hamiltonian_dtype = force_hmltn_dtype

    @property
    def positive_spins(self):
        return self._positive_spins

    @property
    def negative_spins(self):
        return self._negative_spins

    @property
    def electrons(self):
        return self._electrons

    @property
    def side(self):
        return self._side

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def basis_dtype(self):
        return self._basis_dtype

    @property
    def spin_type(self):
        return self._spin_type

    @property
    def basis(self):
        return self._basis

    @property
    def hamiltonian(self):
        return self._hamiltonian

    def basis_vector_index(self, vector):
        """Find the index of a vector."""
        # The try-except block and if statement add 20% computing time. Consider creating a trusted version without
        # the two blocks if considering optimising this function.
        try:
            if vector.dtype == self._basis_dtype:
                return self._inverse_basis_dict[vector.tobytes()]
            return self._inverse_basis_dict[vector.astype(self._basis_dtype).tobytes()]
        except AttributeError:      # Handle tuples, lists etc.
            return self._inverse_basis_dict[np.array(vector, dtype=self._basis_dtype).tobytes()]

    def trusted_basis_vector_index(self, vector):
        """Find the index of a vector, assuming vector is a numpy array of the same dtype as the basis."""
        return self._inverse_basis_dict[vector.tobytes()]

    def _generate_hamiltonian(self):
        bit_index_arr = np.array([2 ** n for n in range(self._grid_size)], dtype=self._basis_dtype)
        model_is_constrained = self._spin_type == "constrained"
        bits_between_bit_pair_dict = _LUTM.get_bits_between_bit_pair_dict(self._basis_dtype)

        def spinless_hops(subvec, other_subvec=None):
            electron_array = np.bitwise_and(subvec, bit_index_arr)
            bool_electron_array = electron_array.astype(bool)
            hop_array = _LUTM.get_square_hop_lut(self._side)[bool_electron_array]   # All hops, even if they collide
            electron_array = electron_array[bool_electron_array]    # Remove zeros
            # Remove colliding hops:
            hop_array = np.bitwise_and(hop_array, np.bitwise_not(subvec, dtype=self._basis_dtype))
            if model_is_constrained:        # Ensure no electron jumps onto a taken slot
                hop_array = np.bitwise_and(hop_array, np.bitwise_not(other_subvec, dtype=self._basis_dtype))
            # Break apart into individual hops:
            hop_array = np.bitwise_and(np.expand_dims(hop_array, 1), bit_index_arr)
            hop_array = np.where(hop_array != 0, np.expand_dims(subvec - electron_array, 1) + hop_array, 0)
            hop_array = hop_array[hop_array != 0]
            parity_masks = [bits_between_bit_pair_dict[i] for i in np.bitwise_xor(hop_array, subvec)]
            parity_masks = np.array(parity_masks, dtype=self._basis_dtype)  # Necessary for empty vectors
            # TODO: When numpy 2.0 comes out, convert this to numpy.bit_count
            parity = np.array([(i.bit_count() % 2) * 2 - 1 for i in np.bitwise_and(parity_masks, subvec)])
            return hop_array, parity

        def hmltn_column(vector):
            plus_hops = spinless_hops(vector[0], vector[1])     # List of hopped vectors when positive electrons hop
            minus_hops = spinless_hops(vector[1], vector[0])    # List of hopped vectors when negative electrons hop

            plus_vectors = np.empty((plus_hops[0].shape[0], 2), dtype=self._basis_dtype)
            plus_vectors[:, 0] = plus_hops[0]
            plus_vectors[:, 1] = vector[1]  # This now contains vectors in bit pair representation

            minus_vectors = np.empty((minus_hops[0].shape[0], 2), dtype=self._basis_dtype)
            minus_vectors[:, 1] = minus_hops[0]
            minus_vectors[:, 0] = vector[0]

            all_vecs = np.concatenate((plus_vectors, minus_vectors))
            all_signs = np.concatenate((plus_hops[1], minus_hops[1]))
            all_vec_indices = [self.trusted_basis_vector_index(new_vec) for new_vec in all_vecs]
            return all_vec_indices, all_signs

        if self._hamiltonian_is_sparse:
            hmltn = sparse.dok_matrix((self._basis.shape[0], self._basis.shape[0]), dtype=self._hamiltonian_dtype)
        else:
            hmltn = np.zeros((self._basis.shape[0], self._basis.shape[0]), dtype=self._hamiltonian_dtype)

        for row_idx, vec in enumerate(self._basis):
            col_indices, values = hmltn_column(vec)
            hmltn[col_indices, row_idx] = values

        if self._hamiltonian_is_sparse:
            hmltn = hmltn.tocsr(copy=False)

        self._hamiltonian = hmltn
        self._hamiltonian.flags.writeable = False


class FreeModel(_SquareModel):
    def __init__(self, side_length, electrons, positive_spins, force_dtype=None,
                 use_sparse_hmltn=False, force_hmltn_dtype=None):
        super().__init__(side_length, electrons, positive_spins, force_dtype,
                         use_sparse_hmltn, force_hmltn_dtype)
        self._spin_type = "free"
        self._basis = generate_free_basis(self.grid_size, self.electrons,
                                          self.positive_spins, dtype=self._basis_dtype)
        self._basis.flags.writeable = False
        self._inverse_basis_dict = _inverse_basis_dict(self._basis)
        self._generate_hamiltonian()


class ConstrainedModel(_SquareModel):
    def __init__(self, side_length, electrons, positive_spins, force_dtype=None,
                 use_sparse_hmltn=False, force_hmltn_dtype=None):
        super().__init__(side_length, electrons, positive_spins, force_dtype,
                         use_sparse_hmltn, force_hmltn_dtype)
        self._spin_type = "constrained"
        if self._positive_spins + self._negative_spins > self._electrons:
            raise ValueError("Constrained model can only have one electron per hole!")
        self._basis = generate_constrained_basis(self.grid_size, self.electrons,
                                                 self.positive_spins, dtype=self._basis_dtype)
        self._basis.flags.writeable = False
        self._inverse_basis_dict = _inverse_basis_dict(self._basis)
        self._generate_hamiltonian()


def generate_free_basis(grid_size, electrons, positive_spins, dtype=np.uint32):
    if grid_size <= 16:
        if positive_spins > electrons:
            raise ValueError("Number of positive spins cannot be higher than number of electrons!")
        spinless_plus = _LUTM.bit_number_dict[positive_spins]
        spinless_minus = _LUTM.bit_number_dict[electrons - positive_spins]
        spinless_plus = spinless_plus[spinless_plus < 2 ** grid_size]
        spinless_minus = spinless_minus[spinless_minus < 2 ** grid_size]
        positive, negative = np.meshgrid(spinless_plus, spinless_minus, indexing='ij')
        positive = positive.flatten()
        negative = negative.flatten()
        return np.stack((positive, negative), axis=1, dtype=dtype)

    raise ValueError("Bases for grid sizes above 16 (4x4) are not implemented yet.")    # TODO: Implement this
    # This will require combinations of 16-bit sets


def generate_constrained_basis(grid_size, electrons, positive_spins, dtype=np.uint32):
    free_basis = generate_free_basis(grid_size, electrons, positive_spins, dtype)
    return free_basis[np.bitwise_and(free_basis[:, 0], free_basis[:, 1]) == 0]


def _inverse_basis_dict(basis):
    """Generate a dictionary, which allows for finding a vector's index."""
    return {vector.tobytes(): index for index, vector in enumerate(basis)}
