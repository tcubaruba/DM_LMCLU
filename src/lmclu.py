import numpy as np
import pandas as pd

from src.sub import find_separation


def __norm(val):
    return np.linalg.norm(val)


def __is_in_neighborhood(df, x_index, proximity_threshold, man_origin, man_basis):
    x = df.iloc[x_index].values
    proximity_value = __norm(x - man_origin) ** 2 - __norm(man_basis @ (x - man_origin).T) ** 2
    is_in = proximity_value < proximity_threshold
    return is_in


def __get_neighborhood(df: pd.DataFrame, proximity_threshold, man_origin, man_basis):
    drop_list = []
    for i in range(df.shape[0]):
        drop_list.append(__is_in_neighborhood(df, i, proximity_threshold, man_origin, man_basis))

    for i in range(len(drop_list)):
        if not drop_list[i]:
            df = df.drop(i)
    df_new = df.to_numpy()
    return pd.DataFrame(df_new)


def __remove_clustered_data_rows(df, df_removed):
    np_arr = df.to_numpy()
    np_delete = df_removed.to_numpy()
    np_new = []
    for item in np_arr:
        if item not in np_delete:
            np_new.append(item)
    return pd.DataFrame(np_new)


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

    while len(data) > max_lm_dim:  # optimization max_lm_dim is lowest border
        lm_dim = 1
        data_copy = data.copy()

        for k in range(max_lm_dim):
            while True:
                if len(data_copy) <= max_lm_dim:  # additional exit criteria
                    break

                goodness_threshold, proximity_threshold, man_origin, man_basis = find_separation.find_separation(
                    data_copy.to_numpy(), k + 1, sampling_level)

                if goodness_threshold <= sensitivity_threshold:
                    break

                n_before = data_copy.shape[0]
                data_copy = __get_neighborhood(data_copy, proximity_threshold, man_origin, man_basis)
                n_after = data_copy.shape[0]

                if n_before <= n_after:
                    # additional exit because sometimes the goodness value is not decreasing
                    # because goodness has an absurd value (we think it's the data! pretty sure bout this)
                    break

                lm_dim = k + 1

        if data_copy.shape[0] > 0:
            # a non empty cluster is found:
            clusters.append(data_copy.to_numpy())  # Note: label of cluster := index
            dims.append(lm_dim)
            data = __remove_clustered_data_rows(data, data_copy)
        else:
            # rest is outlier
            clusters.append(data.to_numpy())
            dims.append(lm_dim)
            break

    return clusters, dims
