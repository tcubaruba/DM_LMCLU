import pandas as pd
from sklearn import metrics
from src import lmclu

__K = 2  # max LM dim
__S = 10  # sampling level
__Gamma = 0.5  # sensitivity threshold

# __file_name = "data/vary-density.csv"
__file_name = "data/mouse.csv"
# __file_name = "/Users/natalia.tretiakova/Documents/Informatik/WS19_20/DM/Group Assignments/LMCLU/lmclu/data/vary-density.csv"


def load_data(file_path):
    df = pd.read_csv(file_path, names=['a1', 'a2', 'label'], header=None, delim_whitespace=True, encoding='utf-8')
    data = df[['a1', 'a2']]
    labels = df['label']
    print(df)
    return data, labels


def invoke_lmclus(df, K, S, Gamma):
    clusters, lm_dims = lmclu.run(df, K, S, Gamma)
    return clusters, lm_dims


def get_pred_labels(clusters):
    # todo how to get predicted labels
    pred_labels = ()
    return


def main():
    df, true_labels = load_data(__file_name)
    clusters, lm_dims = invoke_lmclus(df, __K, __S, __Gamma)

    print("clusters: ")
    print(clusters)

    print("lm_dims: ")
    print(lm_dims)

    pred_labels = get_pred_labels(clusters)
    print("\nsklearn metrics evaluation:")
    print(f"{metrics.fowlkes_mallows_score(true_labels, pred_labels)}")


if __name__ == '__main__':
    main()
