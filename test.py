import basis

# TODO:
# -split into main and functions files, perhaps multiple
# -upload to github
# -implement visualisations of results (matrix displays, different basis representations etc)
#	(matplotlib?)
# -Hamiltonians
# -Hamiltonian diagonalization


side_size = 2
number_of_electrons = 3
number_of_positive_spins = 1

number_of_holes = side_size ** 2

electron_number_array = basis.assign_number_to_electron_number(number_of_holes)

print(electron_number_array)

spinless_basis = basis.get_spinless_basis(number_of_electrons, number_of_holes, electron_number_array)

print(spinless_basis)

spin_basis = basis.get_spin_basis(number_of_electrons, number_of_positive_spins, number_of_holes, electron_number_array)

print(spin_basis)
