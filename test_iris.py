import cProfile
import numpy as np

from rock import RockRealClustering
from utils import categorical_to_binary, purity
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    data = np.loadtxt("data/iris.data", dtype=str, delimiter=",", skiprows=0)
    # data = data[[~np.isin('?', row).all() for row in data]] # remove rows with missing data
    labels = np.asarray(data[:, -1], dtype=str)
    integer_labels = np.zeros(labels.shape[0], dtype=int)
    for i, label in enumerate(np.unique(labels), 0):
        integer_labels[labels == label] = i
    data = np.asarray(data[:, :-1], dtype=float)
    data = MinMaxScaler().fit_transform(data)
    profile = cProfile.Profile()
    profile.enable()
    clustering = RockRealClustering(data, 3, theta=0.5, nbr_max_distance=0.45)
    final_clusters = clustering.clusters()
    profile.disable()

    profile.print_stats(sort='time')
    for i, cluster in enumerate(final_clusters, 1):
        print("Cluster no. {},\nlength = {}".format(i, len(cluster.points)))
        print(labels[cluster.points])

    print("Purity = {}".format(purity(final_clusters, integer_labels)))
