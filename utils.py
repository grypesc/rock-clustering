import numpy as np
from sklearn.preprocessing import OneHotEncoder


def purity(clusters, labels):
    pure_points = 0
    for i, cluster in enumerate(clusters, 1):
        counts = np.bincount(labels[cluster.points])
        dominant = np.argmax(counts)
        pure_points += np.sum(labels[cluster.points] == dominant)
    return pure_points / labels.shape[0]


def overlap_coefficient(feature_similarities_and) -> float:
    return np.sum(feature_similarities_and) / feature_similarities_and.shape[1]


def tanimoto_coefficient(feature_similarities_and, feature_similarities_or) -> float:
    numerator = np.sum(feature_similarities_and, axis=1)
    denominator = np.sum(feature_similarities_or, axis=1)
    return numerator / denominator


def transactions_to_binary(data) -> np.array:
    """ For the total number of products k, products must be chars or ints from 1 to k. """
    bin_columns = np.unique(data).shape[0]
    bin_data = np.zeros((data.shape[0], bin_columns), dtype=int)
    for i in range(0, data.shape[0]):
        bin_data[i, np.asarray(data[i, :], dtype=int) - 1] = 1
    return bin_data


def categorical_to_binary(data) -> np.array:
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    return enc.fit_transform(data)


def binary_search(elements, value):
    left, right = 0, len(elements) - 1

    while left <= right:
        middle = (left + right) // 2

        if elements[middle] == value:
            return middle

        if elements[middle] < value:
            left = middle + 1
        elif elements[middle] > value:
            right = middle - 1


def spherical_distance(lat, lon, lat_vector, lon_vector):
    R = 6371.0
    dlon = lon_vector - lon
    dlat = lat_vector - lat
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c