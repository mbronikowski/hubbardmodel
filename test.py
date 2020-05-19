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
number_of_electrons = 1
number_of_positive_spins = 1

number_of_sites = side_size ** 2

spinless_basis = basis.get_spinless_basis(number_of_electrons, number_of_sites)

print(spinless_basis)

constrained_basis = basis.get_constrained_basis(number_of_electrons, number_of_positive_spins, number_of_sites)

print(constrained_basis)
print(constrained_basis[:, 0])
print(basis.get_constrained_vector_index(constrained_basis, [2, 12]))

hamiltonian = basis.spinless_square_hamiltonian(spinless_basis, number_of_electrons, side_size)

print(hamiltonian.todense())
