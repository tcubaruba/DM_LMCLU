import pandas as pd
from lmclu import lmclu

__K = 2
__S = 1
__Gamma = 0

# __file_name = "data/vary-density.csv"
__file_name = "data/mouse.csv"


def load_data(file_path):
    df = pd.read_csv(file_path, names=['a1', 'a2', 'label'], header=None)
    return df


def find_cluster(df, K, S, Gamma):
    clusters, lm_dims = lmclu.lmclu(df, K, S, Gamma)
    return clusters, lm_dims


if __name__ == '__main__':
    df = load_data(__file_name)
    clusters, lm_dims = find_cluster(df, __K, __S, __Gamma)
    print("clusters: ")
    print(clusters)

    print("lm_dims: ")
    print(lm_dims)
    pass
