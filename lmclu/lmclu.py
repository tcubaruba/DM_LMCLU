import numpy as np
import pandas as pd

from find_separation.find_separation import find_separation


def norm(val):
    return np.linalg.norm(val)


def is_in_neighborhood(df, x_index, proximity_threshold, man_origin, man_basis):
    x = df.iloc[x_index].values
    is_in = norm(x - man_origin) ** 2 - norm(man_basis.T(x - man_origin)) ** 2 < proximity_threshold
    return is_in


def get_neighborhood(df, proximity_threshold, man_origin, man_basis):
    df_copy = df.copy()
    for row_index, row in df.iterrows():
        if not is_in_neighborhood(df, row_index, proximity_threshold, man_origin, man_basis):
            df_copy = df_copy.drop(row_index)
    return df_copy


def lmclu(data: pd.DataFrame, max_lm_dim: int, sampling_level: int, sensitivity_threshold: float) -> (list, list):
    """
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
                goodness_threshold, proximity_threshold, man_origin, man_basis = find_separation(data_copy, k + 1, sampling_level)
                if goodness_threshold <= sensitivity_threshold:
                    break
                data_copy = get_neighborhood(data_copy.to_numpy(), proximity_threshold, man_origin, man_basis)
                lm_dim = k + 1
        # a cluster is found:
        clusters.append(data_copy)  # Note: label of cluster := index
        dims.append(lm_dim)
        data = pd.concat([data, data_copy]).drop_duplicates(keep=False)
    return clusters, dims
