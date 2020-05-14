import basis

# TODO:
# -split into main and functions files, perhaps multiple
# -upload to github
# -implement visualisations of results (matrix displays, different basis representations etc)
#	(matplotlib?)
# -Hamiltonians
# -Hamiltonian diagonalization


sideSize = 2
numberOfElectrons = 3
numberOfPositiveSpins = 1

numberOfHoles = sideSize ** 2

electronNumberArray = basis.assignNumberToElectronNumber(numberOfHoles)

print(electronNumberArray)

spinlessBasis = basis.getSpinlessBasis(numberOfElectrons, numberOfHoles, electronNumberArray)

print(spinlessBasis)

spinBasis = basis.getSpinBasis(numberOfElectrons, numberOfPositiveSpins, numberOfHoles, electronNumberArray)

print(spinBasis)
