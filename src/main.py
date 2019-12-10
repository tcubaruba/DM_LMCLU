import pandas as pd
from sklearn import metrics
from src import lmclu
import numpy as np

__K = 2  # max LM dim
__S = 10  # sampling level
__Gamma = 0.8  # sensitivity threshold

# __file_name = "data/vary-density.csv"
__file_name = "data/mouse.csv"


def load_data(file_path):
    df = pd.read_csv(file_path, header=None, delim_whitespace=True, encoding='utf-8')
    data = df.iloc[:, :-1]
    labels = df[df.shape[1] - 1]
    return data, list(labels.to_numpy())


def invoke_lmclus(df, K, S, Gamma):
    clusters, lm_dims = lmclu.run(df, K, S, Gamma)
    return clusters, lm_dims


def get_pred_labels(data, clusters):
    pred_labels = {}
    nd_data = data.to_numpy()
    for cluster_i in range(len(clusters)):
        cluster = clusters[cluster_i]
        for value in cluster:
            index = int(np.squeeze(np.where(np.all(nd_data == value, axis=1))))
            pred_labels[index] = cluster_i

    return list(pred_labels.values())


def main():
    data, true_labels = load_data(__file_name)
    clusters, lm_dims = invoke_lmclus(data, __K, __S, __Gamma)

    print("clusters: ")
    print(clusters)

    print("lm_dims: ")
    print(lm_dims)

    pred_labels = get_pred_labels(data, clusters)
    print("\nsklearn metrics evaluation:")
    print(f"{metrics.fowlkes_mallows_score( true_labels, pred_labels)}")


if __name__ == '__main__':
    main()
