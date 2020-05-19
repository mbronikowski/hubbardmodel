import basis
import plotting

# TODO:
# -implement visualisations of results (matrix displays, different basis representations etc)
# (matplotlib?)
# -Hamiltonians
# -Convert assign_number_to_electron_number and its use to a private method
# -Hamiltonian diagonalization
# For hamiltonian: https://docs.scipy.org/doc/scipy/reference/sparse.html


side_size = 2
number_of_electrons = 3
number_of_positive_spins = 2

number_of_sites = side_size ** 2

spinless_basis = basis.get_spinless_basis(number_of_electrons, number_of_sites)

print(spinless_basis)

constrained_basis = basis.get_constrained_basis(number_of_electrons, number_of_positive_spins, number_of_sites)
free_basis = basis.get_free_basis(number_of_electrons, number_of_positive_spins, number_of_sites)

print(free_basis)

print(basis.get_spin_vector_index(free_basis, [5, 4]))

# hamiltonian = basis.spinless_square_hamiltonian(spinless_basis, number_of_electrons, side_size)
# print(hamiltonian.todense())
