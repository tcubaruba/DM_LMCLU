from src.sub import get_minimum_error_threshold as err_th
import numpy as np

__infinity = 10000000
__epsilon = 0.00001


def __get_n_random_sample_indices(data, n):
    """return n random row indices of the data set"""
    row_indices = np.random.choice(data.shape[0], n, replace=False)
    return row_indices


def __form_orthonormal_basis(M, O_index):
    space = np.delete(M, O_index, axis=0)
    q, r = np.linalg.qr(space)
    return q


def __make_histogram(distances):
    H, _ = np.histogram(distances)
    return H


def find_separation(D, K, S):
    """
    :param D: dataset
    :param K: max LM dim
    :param S: sampling level
    :return:
        gamma (= goodness threshold), tau (= proximity threshold), mean (= man origin), beta (= man basis)
    """
    gamma = -__infinity
    tau = -__infinity
    mean = 0
    beta = 0

    n = np.min([np.log(__epsilon) / np.log(1 - (1 / S) ** K), D.shape[0]])
    n = int(n)

    for i in range(0, n):
        M_indices = __get_n_random_sample_indices(D, K + 1)
        M = D[M_indices, :]
        O_indices = np.random.choice(M_indices, 1)
        O = D[O_indices, :]
        B = __form_orthonormal_basis(M, O_indices)  # fixed
        B_squeezed = np.squeeze(B)
        distances = []
        for row in range(D.shape[0]):
            x = D[row]
            if x not in M:
                x_new = x - O
                # fixme: (thomas) here's raised an error sometimes: dependent on parameter:
                distances.append(np.linalg.norm(x_new) - np.linalg.norm(x_new @ B_squeezed.T))
        H = __make_histogram(distances)
        T, G = err_th.min_err_threshold(H)
        if G > gamma:
            gamma = G
            tau = T
            mean = O
            beta = B_squeezed

    return gamma, tau, mean, beta