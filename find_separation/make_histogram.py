import numpy as np
import matplotlib.pyplot as plt
def make_histogram(distances):
    H = np.histogram(distances)
    plt.hist(H)
    plt.show()

    return H

