import find_separation.get_minimum_error_threshold as find_min_threshold
import numpy as np

__infinity = 1000000
__epsilon = 0.00001


def __get_n_random_sample_indices(data, n):
    """return n random row indices of the data set"""
    row_indices = np.random.choice(data.shape[0], n, replace=False)
    return row_indices


def __form_orthonormal_basis(M, O_index):
    space = np.delete(M, O_index, axis=0)  # fixme: raises an exception
    return np.linalg.qr(space)


def __evaluate_goodness_of_separation(T, H):
    G = 0
    # todo: (thomas) not implemented yet!!
    # step 1: calculate discriminability [see paper (6)]
    # step 2: depth = J(tau') - J(tau)
    # step 3: G = discriminablility x depth [see paper (7)]
    return G


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
        # todo: (thomas) I'd appreciate it, if we explain the variables a bit what is M, O, B?!
        M_indices = __get_n_random_sample_indices(D, K + 1)
        M = D[M_indices, :]
        O_indices = np.random.choice(M_indices, 1)
        O = D[O_indices, :]
        B, _ = __form_orthonormal_basis(M, O_indices)  # fixme: (thomas) something is wrong here
        B = np.squeeze(B)
        distances = []
        for row in range(D.shape[0]):
            x = D[row]
            if x not in M:
                x_new = x - O
                distances.append(np.linalg.norm(x_new) - np.linalg.norm(x_new @ B.T))
        H = __make_histogram(distances)
        T = find_min_threshold.min_err_threshold(H)
        G = __evaluate_goodness_of_separation(T, H)
        if G > gamma:
            gamma = G
            tau = T
            mean = O
            beta = B

    return gamma, tau, mean, beta
