import folium
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from utils import categorical_to_binary, purity_scikit
from sklearn.cluster import AgglomerativeClustering

if __name__ == '__main__':
    data = np.loadtxt("data/agaricus-lepiota.data", dtype=str, delimiter=",", skiprows=0)
    for i, neighborhood in enumerate(np.unique(data[:, 0]), 0):
        data[data[:, 0] == neighborhood, 0] = i
    labels = np.asarray(data[:, 0], dtype=int)
    data = np.asarray(data[:, 1:])
    data = categorical_to_binary(data)
    agg = AgglomerativeClustering(n_clusters=21).fit(data[:, :])

    print(purity_scikit(agg.labels_, labels))
    print(metrics.homogeneity_score(labels, agg.labels_))

    print("\n################################################################################\n")

    data = np.loadtxt("data/house-votes-84.data", dtype=str, delimiter=",", skiprows=0)
    # data = data[[~np.isin('?', row).all() for row in data]] # remove rows with missing data
    labels = np.asarray(data[:, 0], dtype=str)
    integer_labels = np.zeros(labels.shape[0], dtype=int)
    for i, label in enumerate(np.unique(labels), 0):
        integer_labels[labels == label] = i
    data = data[:, 1:]

    agg = AgglomerativeClustering(n_clusters=2).fit(categorical_to_binary(data))

    print(purity_scikit(agg.labels_, integer_labels))

    print("\n################################################################################\n")

    data = pd.read_csv('data/AB_NYC_2019.csv', delimiter=',')
    data = data[["neighbourhood_group", "latitude", "longitude"]]
    data = data.to_numpy()
    data = data[:2000, :]
    for i, neighborhood in enumerate(np.unique(data[:, 0]), 0):
        data[data[:, 0] == neighborhood, 0] = i
    data = np.asarray(data, dtype=float)
    labels = data[:, 0]
    data = data[:, 1:]

    # Visualizing data with Folium
    fig = folium.Figure(width=550, height=350)
    map = folium.Map(location=data[0, :], tiles='cartodbpositron', zoom_start=11)
    fig.add_child(map)
    colors = ['green', 'gray', 'orange', 'black', 'blue', 'lightblue', 'lightred', 'darkred', 'darkpurple', 'darkgreen',
              'beige',
              'red', 'lightgreen', 'cadetblue', 'lightgray', 'darkblue', 'pink']

    clustering = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(data)
    labels_unique = np.unique(clustering.labels_)

    for i, label_pred in enumerate(labels_unique, 1):
        points = np.where(clustering.labels_ == label_pred)[0]
        print("Cluster no. {},\nlength = {}".format(i, len(points)))
        if len(points) != 1:
            cluster_color = colors[(i - 1) % len(colors)]
            for point in points:
                folium.Marker(location=data[point, :],
                              icon=folium.Icon(color=cluster_color, icon='circle', prefix='fa-')).add_to(map)
        else:
            for point in points:
                folium.Marker(location=data[point, :],
                              icon=folium.Icon(color="purple", icon='circle', prefix='fa-')).add_to(map)

    print("Purity = {}".format(purity_scikit(clustering.labels_, np.asarray(labels, dtype=int))))
    map.save('docs/nyc_clustering_agglomerative.html')

    print("\n################################################################################\n")

    data = np.loadtxt("data/iris.data", dtype=str, delimiter=",", skiprows=0)
    labels = np.asarray(data[:, -1], dtype=str)
    integer_labels = np.zeros(labels.shape[0], dtype=int)
    for i, label in enumerate(np.unique(labels), 0):
        print("{}: {}".format(label, i))
        integer_labels[labels == label] = i
    data = np.asarray(data[:, :-1], dtype=float)
    data = MinMaxScaler().fit_transform(data)
    clustering = AgglomerativeClustering(n_clusters=3).fit(data)

    print("Purity = {}".format(purity_scikit(clustering.labels_, integer_labels)))