from src.sub import get_minimum_error_threshold as err_th
import numpy as np
import scipy

__infinity = 10000000
__epsilon = 0.00001
__bins = 10


def __get_n_random_sample_indices(data, n):
    """return n random row indices of the data set"""
    row_indices = np.random.choice(data.shape[0], n, replace=False)
    return row_indices


def __form_orthonormal_basis(M, O_index):
    # q = scipy.linalg.orth((M - O).T)  # think this is wrong
    space = np.delete(M, O_index, axis=0)
    q = scipy.linalg.orth(space.T)
    # q1, r1 = np.linalg.qr(space.T)
    return q


def __make_histogram(distances):
    H, class_borders = np.histogram(distances, bins=__bins)
    return H, class_borders


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

    n = np.min([np.log(__epsilon) / np.log(1 - (1.0 / S) ** K), D.shape[0]])
    n = int(n)

    for i in range(0, n):
        M_indices = __get_n_random_sample_indices(D, K + 1)
        M = D[M_indices]
        O_indices = np.random.choice(len(M_indices), 1)
        O = M[O_indices]
        O = np.squeeze(O)
        B = __form_orthonormal_basis(M, O_indices)
        B = np.squeeze(B)
        distances = []
        for row in range(1, D.shape[0]):
            if (row not in M_indices):
                x = D[row]
                x_new = x - O
                # fixme: (thomas) here's raised an error sometimes: dependent on parameter:
                distances.append(np.linalg.norm(x_new) - np.linalg.norm(x_new @ B.T))
        H, class_borders = __make_histogram(distances)
        T, G = err_th.min_err_threshold(H, class_borders)
        if G > gamma:
            # print("gamma: ", gamma, " G: ", G, " T: ", T)
            gamma = G
            tau = T
            mean = O
            beta = B

    return gamma, tau, mean, beta
