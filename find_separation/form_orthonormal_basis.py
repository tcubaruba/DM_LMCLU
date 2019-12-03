import numpy as np


def form_orthonormal_basis(M, O):
    space = np.subtract(M, O)
    return np.linalg.qr(space)


