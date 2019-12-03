import numpy as np


def form_orthonormal_basis(M, O_index):
    space = np.delete(M, O_index, axis=0)
    return np.linalg.qr(space)


