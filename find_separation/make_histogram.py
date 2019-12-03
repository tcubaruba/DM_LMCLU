import numpy as np

def make_histogram(distances):
    H, _ = np.histogram(distances)
    return H

