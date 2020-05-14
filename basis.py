import numpy as np
from scipy import special

def assignNumberToElectronNumber(numberOfBits):
    numberOfPossibleVectors = 2 ** numberOfBits
    result = np.empty(numberOfPossibleVectors, dtype=int)  # using int, max 7x7 array
    for i in range(numberOfPossibleVectors):
        result[i] = bin(i).count("1")
    return result


def getSpinlessBasis(numberOfElectrons, numberOfHoles, electronNumberArray):
    itNumberArray = np.nditer(electronNumberArray, flags=['f_index'])
    resultLen = special.comb(numberOfHoles, numberOfElectrons, exact=True)
    result = np.empty(resultLen, dtype=int)
    itResult = np.nditer(result, flags=['f_index'])
    while not itNumberArray.finished:
        if (itNumberArray[0] == numberOfElectrons):
            result[itResult.index] = itNumberArray.index
            itResult.iternext()
        itNumberArray.iternext()
    return result


def getSpinBasis(numberOfElectrons, numberOfPositiveSpins, numberOfHoles, electronNumberArray):
    numberOfNegativeSpins = numberOfElectrons - numberOfPositiveSpins

    positiveSpinArray = getSpinlessBasis(numberOfPositiveSpins, numberOfHoles, electronNumberArray)
    negativeSpinArray = getSpinlessBasis(numberOfNegativeSpins, numberOfHoles, electronNumberArray)

    resultLen = special.comb(numberOfHoles, numberOfElectrons, exact=True) * special.comb(numberOfElectrons,
                                                                                          numberOfPositiveSpins,
                                                                                          exact=True)
    result = np.empty(shape=(resultLen, 2), dtype=int)

    itResult = np.nditer(result, flags=['c_index'])
    itPositive = np.nditer(positiveSpinArray, flags=['f_index'])
    while not itPositive.finished:
        itNegative = np.nditer(negativeSpinArray, flags=['f_index'])

        while not itNegative.finished:
            if not (itPositive[0] & itNegative[0]):
                result[itResult.index][0] = itPositive[0]
                result[itResult.index][1] = itNegative[0]
                itResult.iternext()
            itNegative.iternext()
        itPositive.iternext()
    return result
    #this mimics the implementation by Mathematica subteam
