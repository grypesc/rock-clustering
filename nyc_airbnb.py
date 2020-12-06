from rock_geographical import RockGeoClustering
from utils import purity
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


if __name__ == '__main__':

    data = pd.read_csv('data/AB_NYC_2019.csv', delimiter=',')
    data = data[["neighbourhood_group", "latitude", "longitude"]]
    data = data.to_numpy()
    data = data[:2000, :]
    for i, neighborhood in enumerate(np.unique(data[:, 0]), 0):
        data[data[:, 0] == neighborhood, 0] = i
    data = np.asarray(data, dtype=float)
    labels = data[:, 0]
    data = data[:, 1:]

    clustering = RockGeoClustering(data, 5, theta=0.8, nbr_max_distance=50)
    final_clusters = clustering.clusters()
    for i, cluster in enumerate(final_clusters, 1):
        print("Cluster no. {},\nlength = {}".format(i, len(cluster.points)))
        print(labels[cluster.points])

    print("Purity = {}".format(purity(final_clusters, np.asarray(labels, dtype=int))))
