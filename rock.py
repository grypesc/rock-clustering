import heapq as heap
import numpy as np

from scipy.sparse import lil_matrix


def overlapping_coefficient(feature_similarities) -> float:
    return np.sum(feature_similarities, axis=1) / feature_similarities.shape[1]


def jaccard_coefficient(feature_similarities) -> float:
    nominator = np.sum(feature_similarities, axis=1)
    denominator = 2 * np.full(shape=(feature_similarities.shape[0]), fill_value=feature_similarities.shape[1]) - nominator
    return nominator / denominator


def transactions_to_categorical(data) -> np.array:
    cat_columns = np.unique(data).shape[0]
    cat_data = np.zeros((data.shape[0], cat_columns), dtype=int)
    for i in range(0, data.shape[0]):
        cat_data[i, np.asarray(data[i, :], dtype=int) - 1] = 1
    return cat_data

class Cluster:
    def __init__(self):


class RockClustering:
    """http://theory.stanford.edu/~sudipto/mypapers/categorical.pdf"""

    def __init__(self, S, k, nbr_threshold=0.5):
        self.S = S
        self.k = k
        self.links = self.compute_links(nbr_threshold)

    def compute_links(self, nbr_threshold) -> lil_matrix:
        neighbors_list = self.find_neighbors(nbr_threshold)
        links = lil_matrix((self.S.shape[0], self.S.shape[0]), dtype=int)
        n_rows, n_col = self.S.shape
        for i in range(0, n_rows):
            i_neighbors = neighbors_list[i][0]
            for j in range(0, i_neighbors.shape[0]):
                for l in range(j + 1, i_neighbors.shape[0]):
                    links[i_neighbors[j], i_neighbors[l]] = links[i_neighbors[j], i_neighbors[l]] + 1
        return links

    def find_neighbors(self, threshold) -> list:
        # Return list of arrays of neighbors
        n_rows, n_col = self.S.shape
        neighbors_list = [None for i in range(0, n_rows)]
        for i in range(0, n_rows):
            feature_similarities = self.S == self.S[i, :]
            feature_similarities[i, :] = False
            similarity = jaccard_coefficient(feature_similarities)
            neighbors_list[i] = np.where(similarity >= threshold)
        return neighbors_list




if __name__ == '__main__':
    data = np.loadtxt("data/test.csv", dtype=str, delimiter=",", skiprows=0)
    clustering = RockClustering(transactions_to_categorical(data), 2, nbr_threshold=0.50)
    print(clustering.links)

