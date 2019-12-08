import numpy as np
import pandas as pd

from src.sub import find_separation


def __norm(val):
    return np.linalg.norm(val)


def __is_in_neighborhood(df, x_index, proximity_threshold, man_origin, man_basis):
    x = df.iloc[x_index].values

    # fixme returns always false da fuck!!!
    proximity_value = __norm(x - man_origin) ** 2 - __norm(man_basis * (x - man_origin).T) ** 2
    is_in = proximity_value < proximity_threshold
    return is_in


def __get_neighborhood(df, proximity_threshold, man_origin, man_basis):
    df_copy = df.copy()
    for row_index, row in df.iterrows():
        if not __is_in_neighborhood(df, row_index, proximity_threshold, man_origin, man_basis):
            df_copy = df_copy.drop(row_index)
    return df_copy


def run(data: pd.DataFrame, max_lm_dim: int, sampling_level: int, sensitivity_threshold: float) -> (list, list):
    """
    LMCLU samples random linear manifolds and finds clusters in it. Hereby the distance histogram is calculated and
    searched for the minimum error threshold. The data is partitioned into two groups the data objects in the
    cluster and everything else. Then the best fitting linear manifold is searched and registered as a cluster.
    The process is started over until all objects are clustered. The last cluster contains all outliers.

    :param data: dataset (D)
    :param max_lm_dim: max LM dim (K)
    :param sampling_level: sampling level (S)
    :param sensitivity_threshold: sensitivity threshold (Gamma)
    :return: clusters, dims
    """
    clusters = []  # list of labeled cluster
    dims = []  # list of intrinsic dimensionalities

    while len(data):
        data_copy = data.copy()
        lm_dim = 1
        for k in range(max_lm_dim):
            while True:
                goodness_threshold, proximity_threshold, man_origin, man_basis = find_separation.find_separation(
                    data_copy.to_numpy(), k + 1, sampling_level)
                if goodness_threshold <= sensitivity_threshold:
                    break
                data_copy = __get_neighborhood(data_copy, proximity_threshold, man_origin, man_basis)
                lm_dim = k + 1
            if data_copy.shape[0] == 0:
                break
        # a cluster is found:
        clusters.append(data_copy)  # Note: label of cluster := index
        dims.append(lm_dim)
        data = pd.concat([data, data_copy]).drop_duplicates(keep=False)
    return clusters, dims
