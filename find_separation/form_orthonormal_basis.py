import numpy as np


def form_orthonormal_basis(M, O):
    space = []
    for row in M:
        if((np.array_equal(row, O))):
            space.append(row)
    return np.linalg.qr(space)


