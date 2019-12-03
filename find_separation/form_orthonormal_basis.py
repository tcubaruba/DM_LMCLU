import numpy as np


def form_orthonormal_basis(M, O):
    space = []
    for row in M:
        if((np.array_equal(row, O))):
            space.append(row)
    # space = np.delete(M, O, axis=0)
    # print(space)
    return np.linalg.qr(space)


