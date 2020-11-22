import heapq
import numpy as np

from scipy.sparse import lil_matrix


def overlapping_coefficient(feature_similarities_and) -> float:
    return np.sum(feature_similarities_and) / feature_similarities_and.shape[1]


def jaccard_coefficient(feature_similarities_and, feature_similarities_or) -> float:
    numerator = np.sum(feature_similarities_and, axis=1)
    denominator = np.sum(feature_similarities_or, axis=1)
    return numerator / denominator


def transactions_to_categorical(data) -> np.array:
    # For total number of products k, products must be chars or ints from 1 to k
    cat_columns = np.unique(data).shape[0]
    cat_data = np.zeros((data.shape[0], cat_columns), dtype=int)
    for i in range(0, data.shape[0]):
        cat_data[i, np.asarray(data[i, :], dtype=int) - 1] = 1
    return cat_data


class Cluster:
    def __init__(self, point):
        self.points = [point]
        self.heap = []

    def add_to_heap(self, point_index):
        heapq.heappush(self.heap, (0, point_index))


class RockClustering:
    """http://theory.stanford.edu/~sudipto/mypapers/categorical.pdf"""

    def __init__(self, S, k, nbr_threshold=0.5):
        self.S = S
        self.k = k
        self.nbr_threshold = nbr_threshold
        self.links = self.compute_links(nbr_threshold)
        self.q = [Cluster(i) for i in range(0, S.shape[0])]
        self.clustering()

    def clustering(self):
        for index in range(0, self.S.shape[0]):
            points_linked = self.links.getrow(index).todok().items()
            for point, _ in points_linked:
                point = point[1] if point[1] != index else point[0]
                heap_tuple = (self.goodness_measure(self.q[index], self.q[point]), point)
                heapq.heappush(self.q[index].heap, heap_tuple)


        print(self.q[0].heap)

    def compute_links(self, nbr_threshold) -> lil_matrix:
        neighbors_list = self.find_neighbors(nbr_threshold)
        links = lil_matrix((self.S.shape[0], self.S.shape[0]), dtype=int)
        n_rows, n_col = self.S.shape
        for i in range(0, n_rows):
            i_neighbors = neighbors_list[i][0]
            for j in range(0, i_neighbors.shape[0] - 1):
                for l in range(j + 1, i_neighbors.shape[0]):
                    links[i_neighbors[j], i_neighbors[l]] = links[i_neighbors[j], i_neighbors[l]] + 1
                    # links[i_neighbors[l], i_neighbors[j]] = links[i_neighbors[l], i_neighbors[j]] + 1
        return links

    def find_neighbors(self, threshold) -> list:
        # Return list of arrays of neighbors
        n_rows, n_col = self.S.shape
        neighbors_list = [None for i in range(0, n_rows)]
        for i in range(0, n_rows):
            feature_similarities_and = np.logical_and(self.S, self.S[i, :])
            feature_similarities_and[i, :] = False
            feature_similarities_or = np.logical_or(self.S, self.S[i, :])
            similarity = jaccard_coefficient(feature_similarities_and, feature_similarities_or)
            neighbors_list[i] = np.where(similarity >= threshold)
        return neighbors_list

    def goodness_measure(self, c1, c2) -> float:
        f_of_theta = (1.0 - self.nbr_threshold) / (1.0 + self.nbr_threshold)
        exponent = 1 + 2*f_of_theta
        numerator = 0
        for i in c1.points:
            for j in c2.points:
                numerator += self.links[i, j]
        denominator = (len(c1.points) + len(c1.points))**exponent - len(c1.points)**exponent - len(c2.points)**exponent
        return (-1) * numerator/denominator


if __name__ == '__main__':
    data = np.loadtxt("data/test.csv", dtype=str, delimiter=",", skiprows=0)
    clustering = RockClustering(transactions_to_categorical(data), 2, nbr_threshold=0.50)

