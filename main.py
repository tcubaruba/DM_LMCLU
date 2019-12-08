import pandas as pd
from lmclu import lmclu

__K = 2  # max LM dim
__S = 2  # sampling level
__Gamma = 0  # sensitivity threshold

# __file_name = "data/vary-density.csv"
__file_name = "data/mouse.csv"


def load_data(file_path):
    df = pd.read_csv(file_path, names=['a1', 'a2', 'label'], header=None, delim_whitespace=True)
    data = df[['a1', 'a2']]
    labels = df['label']
    return data, labels


def invoke_lmclus(df, K, S, Gamma):
    clusters, lm_dims = lmclu.lmclu(df, K, S, Gamma)
    return clusters, lm_dims


def main():
    df, labels = load_data(__file_name)
    clusters, lm_dims = invoke_lmclus(df, __K, __S, __Gamma)

    print("clusters: ")
    print(clusters)

    print("lm_dims: ")
    print(lm_dims)


if __name__ == '__main__':
    main()
