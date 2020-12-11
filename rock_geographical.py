import numpy as np

from utils import spherical_distance
from rock import RockClustering


class RockGeoClustering(RockClustering):
    """ This is adapation of rock clustering to geographical data """

    def __init__(self, S: np.array, k: int, theta=0.5, nbr_max_distance=100):
        """
        :param S: Data set to cluster
        :param k: Number of clusters
        :param theta: Theta parameter from the original paper, should be from 0 to 1
        :param nbr_max_distance: Distance in meters below which to classify points as neighbors
        """
        self.nbr_max_distance = nbr_max_distance
        super().__init__(S, k, theta)

    def find_neighbors(self) -> list:
        n_rows, n_col = self.S.shape
        neighbors_list = [None for i in range(0, n_rows)]
        for i in range(0, n_rows):
            distance_vector = spherical_distance(self.S[i, 0], self.S[i, 1], self.S[:, 0], self.S[:, 1])
            distance_vector[i] = np.inf
            neighbors_list[i] = np.where(distance_vector <= self.nbr_max_distance)
        return neighbors_list
