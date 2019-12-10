
Documentation 'LMCLUS' Algorithm
===

**Linear manifold clustering in high dimensional spaces by stochastic search**
2019W 052311-1 Data Mining
_Documentation of the implementation of the LMCLUS algorithm in python by Fenz, Fuchssteiner, Tretiakova, Zenz

GitHub Source Code: https://github.com/tcubaruba/DM_LMCLU
<div style="page-break-after: always;"></div>
# Table of Contents
[TOC]
<div style="page-break-after: always;"></div>
# Structure

```
.
├── README.MD                                 # Readme
├── data                                      # Data repository / example data
│   ├── mouse.csv                             # mouse clusters
│   ├── vary-density.csv                      # varying densities clusters
│   └── ...
├── literature                                # Literature repository
│   ├── LMCLU_PAPER.pdf                       # underlying paper
│   └── ...
├── src                                       # Source code repository
│   ├── sub                                   # sub routines
│   │   ├── __init__.py
│   │   ├── find_separation.py                # helper functions separation:
│   │   │                                     # (1) __get_n_random_sample_indices()
│   │   │                                     # (2) __form_orthonormal_basis()
│   │   │                                     # (3) __make_histogram()
│   │   │                                     # (4) __find_separation()
│   │   │                                     
│   │   ├── get_minimum_error_threshold.py    # helper functions KI-algorithm:
│   │   │                                     # (1) min_err_threshold()
│   │   │                                     # (2) __evaluate_goodness()
│   │   └── ...
│   ├── __init__.py
│   ├── lmclu.py                              # main routine & helper functions:
│   │                                         # (1) __norm()
│   │                                         # (2) __is_in_neighborhood()
│   │                                         # (3) __get_neighborhood() 
│   │                                         # (4) __remove_clustered_data_rows() 
│   │                                         # (5) lmclu() 
│   │
│   ├── main.py                               # function call and load data
│   │                                         # (1) load_data()
│   │                                         # (2) invoke_lmclus()
│   │                                         # (3) get_pred_labels() 
│   │                                         # (4) plot_data() 
│   │                                         # (5) main() 
│   └── ...
└── ...
```
<div style="page-break-after: always;"></div>
# Pseudo-Code

## Main Algorithm
The Algorithm LMCLUS can be viewed as hierarchical-divisive clustering procedure, executing three levels of iteration and expecting three input parameters: $K$ (upper limit on dimension of linear manifolds in which clusters may be embedded), $S$ (sampling level parameter to determine number of trial manifolds), and $\Gamma$ (sensitivity or 'goodness of separation' threshold).

$\textbf{LMCLUS}$(dataset: $D$, max LM dim: $K$, sampling level: $S$, sensitivity threshold: $\Gamma$)
![LMCLU](https://i.imgur.com/sANuhyg.png =800x380)

## Separation

**FindSeparation**(dataset: $D$, dimension: $k$, sampling level: $S$)
![FindSeparation](https://i.imgur.com/WB4wUWt.png =750x300)

## Kittler-Illingworth minimum-error threshold
**FindMinimumErrorThreshold**(histogram: $H$)
```
FindMinimumErrorThreshold(histogram: H)
    calculate CDF's 
    calculate StandarDeviations
    calculate EstimationError
return [T]
```
with _CDF's_ based on the assumption that the two thresholded subset can be split into two population in order to generate two Gaussian distributions. Based on this assumption, and the _Bayes minimum theorem_, we can therefore calculate thresholds for every possible combination of two populations, and use this results to obtain the threshold which maximizes the conditional probability of a point classified correctly. 


## Goodness of threshold
**EvaluateGoodnessOfSeparation**(threshold: $T$, histogram: $H$)
```
EvaluateGoodnessOfSeparation(T, H)
for all t in T:
    argmax discriminability
    argmin depth
    G := discriminability * depth
return [G]
```
where _discriminability_ is defined by $Discriminability = \frac{(\mu_1(\tau) - \mu_2(\tau))^2}{\sigma_1(\tau)^2 + \sigma_2(\tau)^2}$, and $Depth = J(\tau') - J(\tau)$ being the difference of the _local maxima_ and the _minima_.

# Output

## Mouse data
Original data:
![Mouse_TRUE](https://i.imgur.com/yJDZ2rt.png =600x300)


### Best result
sklearn metrics evaluation: $0.8614023213436416$
$\Gamma = 0.8$
$\epsilon = 0.8$
![Mouse_PRED](https://i.imgur.com/qW6M7x1.png =600x300)

### Second best result
sklearn metrics evaluation: $0.7939528290283916$
$\Gamma = 0.8$
$\epsilon = 0.5$
![Mouse_PRED2](https://i.imgur.com/iOvYKWT.png =600x300)


## Varying density data
Original dataset:
![Density_TRUE](https://i.imgur.com/ItpLIQ0.png =600x300)

### Best result
sklearn metrics evaluation: $0.6953084139441168$
$\Gamma = 0.6$
$\epsilon = 0.8$
![](https://i.imgur.com/1ZgNqOy.png =600x300)

### Second best result
sklearn metrics evaluation: $0.6369120047610677$
$\Gamma = 0.7$
$\epsilon = 0.75$
![Density_PRED](https://i.imgur.com/VEaTKpI.png =600x300)
 


# Problems

## Reproducibility
The results are in general not easily reproducable due to its stochastic foundation.
A couple of iterations had to be made till we could get these results. Since the method is based on stochastic choice of points, the results are somewhat different, e.g. on the one hand only one cluster, or on the other hand about 10-15 clusters were found, which happened quite frequently although same the parameter configuration was used.

Best result for mouse data: ~$0.86$
Best result for density data: ~$0.75$

## Choice of parameters
Since the results are stochastic and in general not easily reproducable 1:1, it is difficult to estimate best values for $\Gamma$, $S$, and $\epsilon$. Greedy search is not optimal as a combination of parameters which was working good one pass, gives  worse results next pass.

The paper also suggests to choose a $\Gamma$ between $0.4$ and $0.5$, though our best results were achieved by $\Gamma$ parameter setting of about $0.8$.

## Cluster separation
Sometimes very small clusters with only 2-3 elements were detected, therefore we implemented an additional break statement if cluster is too small.

## Goodness of Separation
Added additional exit, because sometimes the goodness value is not decreasing (goodness value explodes). Might be a possible data issue, or a faulty implementation.



# References
* Haralick, R., & Harpaz, R. (2005, July). Linear manifold clustering. In _International Workshop on Machine Learning and Data Mining in Pattern Recognition_ (pp. 132-141). Springer, Berlin, Heidelberg.
* Kittler, J., & Illingworth, J. (1986). Minimum error thresholding. _Pattern recognition_, 19(1), 41-47.
* [python package _PyThreshold_](https://pypi.org/project/pythreshold/ "pythreshold")
* [LMCLUS implementation at ELKI](https://elki-project.github.io/releases/current/doc/de/lmu/ifi/dbs/elki/algorithm/clustering/correlation/LMCLUS.html "LMCLUS implementation at ELKI")
