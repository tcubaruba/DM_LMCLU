import numpy as np

__infinity = 10000000


# todo (thomas): if "min_err_threshold(histogram)" function from manuel aquado martinez is not fitting for lmclus (since it is taken from a different context)
# todo (thomas): implement "min_err_threshold_alternative(histogram)" - here is the java code from elki implementation (this should be super fine)
# todo (thomas): but both implementations seem to do the same?!
# todo (thomas): checkout: java_example_implementation/LMCLUS.java [line 385-462]

def min_err_threshold(histogram: np.ndarray):  # todo (thomas) here could be a bug?!
    """
    NOTE: THIS Method is HIGHLY INSPIRED BY THE SOURCE CODE PROVIDED FROM:
    https://github.com/manuelaguadomtz/pythreshold
    author: Manuel Aguado Martínez (2017)

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
    error = 1 + 2 * (error_a - error_b)

    # print("Error: ", error)
    # print("Np. argmin: ", np.argmin(error))

    goodness, best_pos = __evaluate_goodness(f_std, f_mean, b_std, b_mean, error)
    return histogram[best_pos], goodness


def min_err_threshold_alternative(histogram: np.ndarray):
    f_std = []  # := mu1
    b_std = []  # := mu2
    f_mean = []  # := sigma1
    b_mean = []  # := sigma2
    error = []  # := jt

    # # todo (thomas) convert java code to python
    #    private double[] findAndEvaluateThreshold(DoubleDynamicHistogram histogram)
    #       int n = histogram.getNumBins();
    #       double[] p1 = new double[n];
    #       double[] p2 = new double[n];
    #       double[] mu1 = new double[n];
    #       double[] mu2 = new double[n];
    #       double[] sigma1 = new double[n];
    #       double[] sigma2 = new double[n];
    #       double[] jt = new double[n];
    #       // Forward pass
    #       {
    #         MeanVariance mv = new MeanVariance();
    #         DoubleHistogram.Iter forward = histogram.iter();
    #         for(int i = 0; forward.valid(); i++, forward.advance()) {
    #           p1[i] = forward.getValue() + ((i > 0) ? p1[i - 1] : 0);
    #           mv.put(i, forward.getValue());
    #           mu1[i] = mv.getMean();
    #           sigma1[i] = mv.getNaiveStddev();
    #         }
    #       }
    #       // Backwards pass
    #       {
    #         MeanVariance mv = new MeanVariance();
    #         DoubleHistogram.Iter backwards = histogram.iter();
    #         backwards.seek(histogram.getNumBins() - 1); // Seek to last
    #         //
    #         for(int j = n - 1; backwards.valid(); j--, backwards.retract()) {
    #           p2[j] = backwards.getValue() + ((j + 1 < n) ? p2[j + 1] : 0);
    #           mv.put(j, backwards.getValue());
    #           mu2[j] = mv.getMean();
    #           sigma2[j] = mv.getNaiveStddev();
    #         }
    #       }
    #       //
    #       for(int i = 0; i < n; i++) {
    #          jt[i] = 1.0 + 2 * (p1[i] * (FastMath.log(sigma1[i]) - FastMath.log(p1[i])) + p2[i] * (FastMath.log(sigma2[i]) - FastMath.log(p2[i])));
    #       }

    goodness, best_pos = __evaluate_goodness(f_std, f_mean, b_std, b_mean, error)
    return histogram[best_pos], goodness


def __evaluate_goodness(f_std, f_mean, b_std, b_mean, error):  # todo (thomas): here could be a bug?!
    """
    NOTE: THIS METHOD IS HIGHLY INSPIRED FROM THE IMPLEMENTATION IN ELKI:
        author: ELKI Development Team (2019)
        https://elki-project.github.io/releases/
        package de.lmu.ifi.dbs.elki.algorithm.clustering.correlation;

    """
    n = len(error)

    best_pos = -1
    best_goodness = -__infinity
    dev_prev = error[1] - error[0]

    for i in range(n - 1):
        dev_cur = error[i + 1] - error[i]

        if dev_cur >= 0 >= dev_prev:
            # Local minimum found - calculate depth
            lowest_maxima = __infinity
            for j in range(i, 0, -1):
                if error[j - 1] < error[j]:
                    lowest_maxima = min(lowest_maxima, error[j])
                    break

            for j in range(i + 1, n - 2):
                if error[j + 1] < error[j]:
                    lowest_maxima = min(lowest_maxima, error[j])
                    break

            local_depth = lowest_maxima - error[i]

            mean_diff = f_mean[i] - b_mean[i]

            discriminability = mean_diff ** 2 / (f_std[i] ** 2 + b_std[i] ** 2)
            #
            # print("Dicriminability: ", discriminability)
            # print("Depth: ", local_depth)

            goodness = local_depth * discriminability
            if goodness > best_goodness:
                best_goodness = goodness
                best_pos = i

        dev_prev = dev_cur

    return best_goodness, best_pos
