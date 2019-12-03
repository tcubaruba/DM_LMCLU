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
        if not is_in_neighborhood(df, proximity_threshold, man_origin, man_basis):
            df_copy = df_copy.drop(row_index)
    return df_copy


def lmclu(D: pd.Dataframe, K: int, S: int, Gamma: float) -> (list, list):
    """
    :param D: dataset
    :param K: max LM dim
    :param S: sampling level
    :param Gamma: sensitivity threshold
    :return: clusters, dims
    """
    clusters = []  # list of labeled cluster
    dims = []  # list of intrinsic dimensionalities

    while len(D):
        D_copy = D.copy()
        lm_dim = 1
        for k in range(K):
            while True:
                goodness_threshold, proximity_threshold, man_origin, man_basis = find_separation(D_copy, k + 1, S)
                if goodness_threshold <= Gamma:
                    break
                D_copy = get_neighborhood(D_copy.to_numpy(), proximity_threshold, man_origin, man_basis)
                lm_dim = k
        # a cluster is found:
        clusters.append(D_copy)  # Note: label of cluster := index
        dims.append(lm_dim)
        D = pd.concat([D, D_copy]).drop_duplicates(keep=False)
    return clusters, dims
