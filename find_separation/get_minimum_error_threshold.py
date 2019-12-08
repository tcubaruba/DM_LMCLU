"""
Note: the source code in this file is highly inspired by the source code provided from:
    https://github.com/manuelaguadomtz/pythreshold
    author: Manuel Aguado Martínez (2017)

    Reference of the implemented algorithm:
    Kittler, J. and J. Illingworth. ‘‘On Threshold Selection Using Clustering
    Criteria,’’ IEEE Transactions on Systems, Man, and Cybernetics 15, no. 5
    (1985): 652–655.
"""
import numpy as np

__infinity = 1000000


def min_err_threshold(histogram: np.ndarray):
    """
    Runs the minimum error thresholding algorithm.
    :param histogram (numpy ndarray of floats)
    :return: The threshold that minimize the error
    """
    w_backg = histogram.cumsum()
    w_backg[w_backg == 0] = 1
    w_foreg = w_backg[-1] - w_backg
    w_foreg[w_foreg == 0] = 1

    # Cumulative distribution function
    cdf = np.cumsum(histogram * np.arange(len(histogram)))

    # Means (Last term is to avoid divisions by zero)
    b_mean = cdf / w_backg
    f_mean = (cdf[-1] - cdf) / w_foreg

    # Standard deviations
    b_std = ((np.arange(len(histogram)) - b_mean) ** 2 * histogram).cumsum() / w_backg
    f_std = ((np.arange(len(histogram)) - f_mean) ** 2 * histogram).cumsum()
    f_std = (f_std[-1] - f_std) / w_foreg

    # To avoid log of 0 invalid calculations
    b_std[b_std == 0] = 1
    f_std[f_std == 0] = 1

    # Estimating error
    error_a = w_backg * np.log(b_std) + w_foreg * np.log(f_std)
    error_b = w_backg * np.log(w_backg) + w_foreg * np.log(w_foreg)
    error = 1 + 2 * error_a - 2 * error_b

    goodness, best_pos = evaluate_goodness(f_std, f_mean, b_std, b_mean, error)

    return error[best_pos], goodness


def evaluate_goodness(f_std, f_mean, b_std, b_mean, error):
    n = len(error)

    best_pos = -1
    best_goodness = -__infinity
    dev_prev = error[1] - error[0]

    for i in range(n-1):
        dev_cur = error[i + 1] - error[i]

        if dev_cur >= 0 >= dev_prev:
            # Local minimum found - calculate depth
            lowest_maxima = __infinity
            # for (int j = i - 1; j > 0; j--) {
            for j in range(i, 0, -1):
                if error[j - 1] < error[j]:
                    lowest_maxima = min(lowest_maxima, error[j])
                    break

            # for (int j = i + 1; j < n - 2; j++) {
            for j in range(i + 1, n - 2):
                if error[j + 1] < error[j]:
                    lowest_maxima = min(lowest_maxima, error[j])
                    break

            local_depth = lowest_maxima - error[i]

            mean_diff = f_mean[i] - b_mean[i]

            discriminability = mean_diff ** 2 / (f_std[i] ** 2 + b_std[i] ** 2)

            goodness = local_depth * discriminability
            if goodness > best_goodness:
                best_goodness = goodness
                best_pos = i

        dev_prev = dev_cur

    return best_goodness, best_pos
