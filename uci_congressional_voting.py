import cProfile
import numpy as np

from rock import RockClustering
from utils import categorical_to_binary, purity

if __name__ == '__main__':
    data = np.loadtxt("data/house-votes-84.csv", dtype=str, delimiter=",", skiprows=0)
    # data = data[[~np.isin('?', row).all() for row in data]] # remove rows with missing data
    labels = np.asarray(data[:, 0], dtype=str)
    integer_labels = np.zeros(labels.shape[0], dtype=int)
    for i, label in enumerate(np.unique(labels), 0):
        integer_labels[labels == label] = i
    data = data[:, 1:]

    profile = cProfile.Profile()
    profile.enable()
    clustering = RockClustering(categorical_to_binary(data), 2, nbr_threshold=0.73)
    final_clusters = clustering.clusters()
    profile.disable()

    profile.print_stats(sort='time')
    for i, cluster in enumerate(final_clusters, 1):
        print("Cluster no. {},\nlength = {}".format(i, len(cluster.points)))
        print(labels[cluster.points])

    print("Purity = {}".format(purity(final_clusters, integer_labels)))
