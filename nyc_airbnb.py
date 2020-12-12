from rock_geographical import RockGeoClustering
from utils import purity
import folium
import numpy as np
import pandas as pd  # nyc_airbnb requires smarter loading then numpy loading

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

    # Creating Basemap
    fig = folium.Figure(width=550, height=350)
    map = folium.Map(location=data[0, :], tiles='cartodbpositron', zoom_start=11)
    fig.add_child(map)
    colors = ['green', 'gray', 'orange', 'black', 'blue', 'lightblue', 'lightred', 'darkred', 'darkpurple', 'darkgreen', 'beige',
              'red', 'lightgreen', 'cadetblue', 'lightgray',  'darkblue', 'pink']

    clustering = RockGeoClustering(data, 5, theta=0.5, nbr_max_distance=50)
    final_clusters = clustering.clusters()
    for i, cluster in enumerate(final_clusters, 1):
        print("Cluster no. {},\nlength = {}".format(i, len(cluster.points)))
        print(labels[cluster.points])
        counts = np.bincount(np.asarray(labels, dtype=int)[cluster.points])
        dominant = np.argmax(counts)
        if len(cluster.points) != 1:
            cluster_color = colors.pop(0)
            colors.append(cluster_color)
            for point in cluster.points:
                folium.Marker(location=data[point, :],
                              icon=folium.Icon(color=cluster_color, icon='circle', prefix='fa-')).add_to(map)
        else:
            for point in cluster.points:
                folium.Marker(location=data[point, :],
                              icon=folium.Icon(color="purple", icon='circle', prefix='fa-')).add_to(map)

    print("Purity = {}".format(purity(final_clusters, np.asarray(labels, dtype=int))))

    map.save('docs/nyc_clustering.html')
