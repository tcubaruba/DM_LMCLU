import pandas as pd
from sklearn import metrics
from src import lmclu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # to make plots

__K = 2  # max LM dim
__S = 5  # sampling level
__Gamma = 0.8  # sensitivity threshold

__file_name = "data/vary-density.csv"
# __file_name = "data/mouse.csv"



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


def plot_data(data, labels, convert = False):
    if(convert):
        data = data.astype('float64')
        labels_unique = np.unique(labels)
        dict = {}
        i = 0
        for label in labels_unique:
            dict.update({label: i})
            i += 1

        new_labels = [dict[label] for label in labels]
    else:
        new_labels = labels

    df = pd.DataFrame(data)
    df["label"] = new_labels

    sns.pairplot(data=df, x_vars=[0], y_vars=[1], hue="label")
    plt.show()


def main():
    data, true_labels = load_data(__file_name)
    plot_data(data, np.array(true_labels), convert=True)
    clusters, lm_dims = invoke_lmclus(data, __K, __S, __Gamma)

    print("clusters: ")
    print(clusters)

    print("lm_dims: ")
    print(lm_dims)

    pred_labels = get_pred_labels(data, clusters)
    plot_data(data, pred_labels)
    print("\nsklearn metrics evaluation:")
    print(f"{metrics.fowlkes_mallows_score(true_labels, pred_labels)}")


if __name__ == '__main__':
    main()
