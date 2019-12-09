from src.sub import get_minimum_error_threshold as err_th
import numpy as np
import scipy

__infinity = 10000000
__epsilon = 0.00001


def __get_n_random_sample_indices(data, n):
    """return n random row indices of the data set"""
    row_indices = np.random.choice(data.shape[0], n, replace=False)
    return row_indices


def __form_orthonormal_basis(M, O_index):
    space = np.delete(M, O_index, axis=0)
    # q, r = np.linalg.qr(space)
    q = scipy.linalg.orth(space)
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
        O_indices = np.random.choice(len(M_indices), 1)
        O = D[O_indices, :]
        O = np.squeeze(O)
        B = __form_orthonormal_basis(M, O_indices)  # fixed
        distances = []
        for row in range(1, D.shape[0]):
            if(row not in M_indices):
                x = D[row]
                x_new = x - O
                # print("X new: ", x_new)
                # print("X_new @ B: ", x_new @ B.T)
                # print("Norm x: ", np.linalg.norm(x_new))
                # print("Norm x@ B", np.linalg.norm(B@x_new))
                # fixme: (thomas) here's raised an error sometimes: dependent on parameter:
                distances.append(np.linalg.norm(x_new) - np.linalg.norm(x_new @ B.T))
        # print(distances)
        H = __make_histogram(distances)
        # print("Histogram: ", H)
        T, G = err_th.min_err_threshold(H)
        if G > gamma:
            # print("gamma: ", gamma, " G: ", G, " T: ", T)
            gamma = G
            tau = T
            mean = O
            beta = B

    return gamma, tau, mean, beta
