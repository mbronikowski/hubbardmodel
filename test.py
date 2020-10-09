import matplotlib.pyplot as plt
import hubbard
import plotting
from scipy.sparse import linalg
import numpy as np


# TODO:
# -implement visualisations of results (matrix displays, different basis representations etc)
# (matplotlib?)
# -Hamiltonians
# -Convert assign_number_to_electron_number and its use to a private method
# -Hamiltonian diagonalization
# For hamiltonian: https://docs.scipy.org/doc/scipy/reference/sparse.html


side_size = 3
number_of_electrons = 3
number_of_positive_spins = 2
number_of_negative_spins = number_of_electrons - number_of_positive_spins

number_of_sites = side_size ** 2

hubbard.initialize_square_model(side_size, number_of_electrons, number_of_positive_spins)
hubbard.initialize_bases()


spinless_basis = hubbard._generate_spinless_basis(number_of_electrons, number_of_sites)
spinless_basis_plus_1 = hubbard._generate_spinless_basis(number_of_electrons + 1, number_of_sites)

# free_basis = hubbard._generate_free_basis(number_of_electrons, number_of_positive_spins, number_of_sites)
# print(free_basis)

constrained_basis = hubbard._generate_constrained_basis(number_of_electrons, number_of_positive_spins, number_of_sites)
# free_basis = hubbard._generate_free_basis(number_of_electrons, number_of_positive_spins, number_of_sites)

print(constrained_basis)

# print(free_basis)

# free_hmltn = hubbard.free_square_hamiltonian(free_basis, number_of_electrons, number_of_positive_spins, side_size)

# constr_hmltn = hubbard.constrained_square_hamiltonian(constrained_basis, number_of_electrons, number_of_positive_spins, side_size)

# plt.spy(constr_hmltn, ms=1)
# plt.show()


# hamiltonian = hubbard.spinless_square_hamiltonian(spinless_basis, number_of_electrons, side_size)
# print(hamiltonian.todense())



# print(spinless_basis.shape)
# hamiltonian = hubbard.free_square_hamiltonian(free_basis, number_of_electrons, number_of_positive_spins, side_size)
#
# vec = np.ones(free_basis.shape[0])
# print(vec)
# result = hubbard._multiply_vec_by_free(vec)
# print(result)
# print(hamiltonian.dot(vec))
# print(result - hamiltonian.dot(vec))
#
# hamiltonian_linear_operator = linalg.LinearOperator((free_basis.shape[0], free_basis.shape[0]),
#                                                     matvec=hubbard._multiply_vec_by_free)
#
# print("eigensystem")
# print(linalg.eigsh(hamiltonian_linear_operator, k=125))
#
# print(linalg.eigsh(hamiltonian, k=125))

# print(hubbard._generate_free_basis(number_of_electrons, number_of_positive_spins, number_of_sites).shape)

print(spinless_basis)
print(spinless_basis_plus_1)