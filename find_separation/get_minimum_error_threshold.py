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
    cdf = np.cumsum(histogram * np.arange(len(histogram)))
    b_mean = cdf / w_backg
    f_mean = (cdf[-1] - cdf) / w_foreg
    b_std = ((np.arange(len(histogram)) - b_mean) ** 2 * histogram).cumsum() / w_backg
    f_std = ((np.arange(len(histogram)) - f_mean) ** 2 * histogram).cumsum()
    f_std = (f_std[-1] - f_std) / w_foreg
    b_std[b_std == 0] = 1
    f_std[f_std == 0] = 1
    error_a = w_backg * np.log(b_std) + w_foreg * np.log(f_std)
    error_b = w_backg * np.log(w_backg) + w_foreg * np.log(w_foreg)
    error = 1 + 2 * error_a - 2 * error_b

    return np.argmin(error)
