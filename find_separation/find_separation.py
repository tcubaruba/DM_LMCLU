import find_separation.evaluate_goodness_of_separation as evaluate
import find_separation.find_minimum_error_threshold as find_min_threshold
import find_separation.form_orthonormal_basis as form_basis
import find_separation.make_histogram as make_hist

import numpy as np

__infinity = 1000000
__epsilon = 0.00001


def find_separation(D, K, S):
    """
    :param D: dataset
    :param K: max LM dim
    :param S: sampling level
    :return:
        gamma: goodness_threshold
        tau: proximity_threshold
        mean: man_origin
        beta: man_basis
    """
    gamma = -__infinity
    tau = -__infinity
    mean = 0
    beta = 0

    n = np.min([np.log(__epsilon) / np.log(1 - (1 / S) ** K), D.shape[0]])
    n = int(n)

    for i in range(0, n):
        # todo: (thomas) I'd appreciate it, if we explain the variables a bit what is M, O, B?!
        M_indexes = find_random_point(D, K + 1)
        M = D[M_indexes, :]
        O_index = np.random.choice(M_indexes, 1)
        O = D[O_index, :]
        B, _ = form_basis.form_orthonormal_basis(M, O_index)  # fixme: (thomas) something is wrong here
        B = np.squeeze(B)
        distances = []
        for row in range(D.shape[0]):
            x = D[row]
            if x not in M:
                x_new = x - O
                distances.append(np.linalg.norm(x_new) - np.linalg.norm(x_new @ B.T))
        H = make_hist.make_histogram(distances)
        T = find_min_threshold.min_err_threshold(H)
        G = evaluate.evaluate_goodness_of_separation(T, H)
        if G > gamma:
            gamma = G
            tau = T
            mean = O
            beta = B

    return gamma, tau, mean, beta


def find_random_point(dataset, n_points):
    rows = np.random.choice(dataset.shape[0], n_points, replace=False)
    return rows
