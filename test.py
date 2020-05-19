import scipy.sparse
import numpy as np
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
number_of_electrons = 2
number_of_positive_spins = 1

number_of_sites = side_size ** 2

spinless_basis = basis.get_spinless_basis(number_of_electrons, number_of_sites)

print(spinless_basis)

spin_basis = basis.get_spin_basis(number_of_electrons, number_of_positive_spins, number_of_sites)

#print(spin_basis)

#print (spin_basis[:,0])

#print(basis.get_spin_vector_index(spin_basis, [2, 12]))

print(basis.get_square_electron_hop_lookup())

hops = basis.list_possible_spinless_square_hops(5, 2, 3)

print(hops)

basis.spinless_square_hamiltonian(spinless_basis, number_of_electrons, side_size)

hamiltonian = sparse.dok_matrix((spinless_basis.size(), spinless_basis.size()), dtype=np.double)
