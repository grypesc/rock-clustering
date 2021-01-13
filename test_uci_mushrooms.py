import cProfile

import numpy as np

from rock import RockClustering
from utils import categorical_to_binary, purity

if __name__ == '__main__':
    data = np.loadtxt("data/agaricus-lepiota.data", dtype=str, delimiter=",", skiprows=0)
    for i, neighborhood in enumerate(np.unique(data[:, 0]), 0):
        data[data[:, 0] == neighborhood, 0] = i
    labels = np.asarray(data[:, 0], dtype=int)
    data = np.asarray(data[:, 1:])

    profile = cProfile.Profile()
    profile.enable()
    clustering = RockClustering(categorical_to_binary(data[:, :]), 20, theta=0.80)
    final_clusters = clustering.clusters()
    profile.disable()
    profile.print_stats(sort='cumulative')

    for i, cluster in enumerate(final_clusters, 1):
        print("Cluster no. {},\nlength = {}".format(i, len(cluster.points)))
        print(labels[cluster.points])

    print("Purity = {}".format(purity(final_clusters, labels[:])))
