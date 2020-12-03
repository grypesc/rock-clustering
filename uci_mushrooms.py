from rock import RockClustering, categorical_to_binary, transactions_to_binary, purity
import numpy as np

if __name__ == '__main__':
    data = np.loadtxt("data/mushrooms.csv", dtype=str, delimiter=",", skiprows=0)
    labels = np.asarray(data[:, -1], dtype=int)
    data = data[:, :-1]

    clustering = RockClustering(categorical_to_binary(data[:1000, :]), 20, nbr_threshold=0.80)
    final_clusters = clustering.clusters()
    for i, cluster in enumerate(final_clusters, 1):
        print("Cluster no. {},\nlength = {}".format(i, len(cluster.points)))
        print(labels[cluster.points])

    print("Purity = {}".format(purity(final_clusters, labels[:1000])))
