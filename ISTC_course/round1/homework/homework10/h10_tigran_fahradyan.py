import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

def create_random_dist(n_dist, range=(0, 100)):
    return np.random.uniform(*range, (30, n_dist)).round().T


dists = create_random_dist(10)

def mean_values_from_dist(dists, n_values, iters_count=10000):
    return [np.random.choice(dist, n_values).mean() for _ in range(iters_count) for dist in dists]

values = mean_values_from_dist(dists, 100)
plt.hist(values);
