import matplotlib.pyplot as plt
import basis
import plotting
from scipy.sparse.linalg import eigsh


# TODO:
# -implement visualisations of results (matrix displays, different basis representations etc)
# (matplotlib?)
# -Hamiltonians
# -Convert assign_number_to_electron_number and its use to a private method
# -Hamiltonian diagonalization
# For hamiltonian: https://docs.scipy.org/doc/scipy/reference/sparse.html


side_size = 3
number_of_electrons = 4
number_of_positive_spins = 3
number_of_negative_spins = number_of_electrons - number_of_positive_spins

number_of_sites = side_size ** 2

spinless_basis = basis.get_spinless_basis(number_of_electrons, number_of_sites)

print(spinless_basis)

constrained_basis = basis.get_constrained_basis(number_of_electrons, number_of_positive_spins, number_of_sites)
free_basis = basis.get_free_basis(number_of_electrons, number_of_positive_spins, number_of_sites)

print(constrained_basis)


# print(free_basis)

free_hmltn = basis.free_square_hamiltonian(free_basis, number_of_electrons, number_of_positive_spins, side_size)

constr_hmltn = basis.constrained_square_hamiltonian(constrained_basis, number_of_electrons, number_of_positive_spins, side_size)

plt.spy(constr_hmltn, ms=1)
plt.show()

# print(eigsh(free_hmltn, k=20))

# hamiltonian = basis.spinless_square_hamiltonian(spinless_basis, number_of_electrons, side_size)
# print(hamiltonian.todense())
