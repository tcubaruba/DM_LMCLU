import find_separation.evaluate_goodness_of_separation as evaluate
import find_separation.find_minimum_error_threshold as find_min_threshold
import find_separation.form_orthonormal_basis as form_basis
import find_separation.make_histogram as make_hist

import numpy as np
import random

# d: dataset, k: dimension, s: sampling level


def find_separation(D, K, S):
    infinity = 1000000
    epsilon = 0.00001
    gamma = -infinity
    tau = -infinity
    mean = 0
    beta = 0

    n = np.min([np.log(epsilon)/np.log(1-(1/S)**K), D.shape[0]])
    n = int(n)

    for i in range(0, n):
        M = find_random_point(D, K+1)
        O = find_random_point(M, 1)
        B = form_basis.form_orthonormal_basis(M, O)
        distances = []
        for row in range(D.shape[0]):
            x = D[row]
            if (x not in M):
                x_new = x - O
                distances.append(np.linalg.norm(x_new) - np.linalg.norm(B.T@x_new))
        H = make_hist.make_histogram(distances)
        T = find_min_threshold.find_minimum_error_threshold(H)
        G = evaluate.evaluate_goodness_of_separation(T, H)
        if(G>gamma):
            gamma = G
            tau = T
            mean = O
            beta = B

    return gamma, tau, mean, beta


def find_random_point(dataset, n_points):
    rows = np.random.choice(dataset.shape[0], n_points)
    points_array = dataset[rows, :]
    return points_array
