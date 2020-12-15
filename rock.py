import copy
import heapq
import numpy as np

from utils import tanimoto_coefficient


class Cluster:
    def __init__(self, points, heap=None):
        """
        :param points: Points belonging to this cluster
        :param heap: Local heap
        """
        self.points = points
        self.heap = []
        if heap is not None:
            self.heap = heap

    def __lt__(self, other):
        """Comparing clusters according to goodness measure of their heap roots.
        If cluster is outlying and heap is empty than we want it to go down in the global heap Q"""
        if self.heap and other.heap:
            return self.heap[0][0] < other.heap[0][0]
        elif other.heap:
            return 0
        return 1


class RockClustering:
    """http://theory.stanford.edu/~sudipto/mypapers/categorical.pdf"""
    """ Many variable names used in this implementation correspond to the original paper linked above,
     please take a look before you start reading the code. In case of lack of memory, change links matrix to
      sparse matrix (Scipy lil_matrix) but this will slow down the algorithm."""

    def __init__(self, S: np.array, k: int, nbr_threshold=0.5):
        """
        :param S: Data set to cluster
        :param k: Number of clusters to find
        :param nbr_threshold: Theta parameter from the original paper, should be from 0 to 1
        """
        self.S = S
        self.k = k
        self.nbr_threshold = nbr_threshold
        f_of_theta = (1.0 - self.nbr_threshold) / (1.0 + self.nbr_threshold)
        self.goodness_exponent = 1 + 2 * f_of_theta  # Used for goodness of measure calculation
        self.links = self.compute_links()
        self.q = [Cluster([i]) for i in range(0, S.shape[0])]
        self.global_heap = []  # In the paper it is Q
        self.clustering()

    def clustering(self) -> None:
        """Performs iterative cluster merging, O(n^3) worst case, can be n^2log(n) if you use linked lists to store a heap"""
        for index in range(0, self.S.shape[0]):
            # Build local heaps
            points_linked = np.nonzero(self.links[index, :])[0]
            for point in points_linked:
                heap_tuple = (self.goodness_measure(self.q[index], self.q[point]), self.q[point])
                heapq.heappush(self.q[index].heap, heap_tuple)

        self.global_heap = copy.copy(self.q)
        heapq.heapify(self.global_heap)

        while len(self.global_heap) > self.k:
            #  Merge clusters until you reach k clusters or local heaps are empty and ROCK can't continue
            if not self.global_heap[0].heap:
                print("Could not find more clusters to merge. Stopped at {} clusters".format(len(self.global_heap)))
                break
            u = heapq.heappop(self.global_heap)
            v = u.heap[0][1]
            self.global_heap.remove(v)
            w = Cluster(sorted([*u.points, *v.points]))
            self.q[max(u.points[0], v.points[0])] = None
            self.q[w.points[0]] = w
            nbr_clusters = set([*[j for i, j in u.heap], *[j for i, j in v.heap]])
            nbr_clusters.remove(u)
            nbr_clusters.remove(v)

            for x in nbr_clusters:
                #  Update local heaps of neighbors of merged cluster u and v
                self.links[x.points[0], w.points[0]] = self.links[x.points[0], u.points[0]] + self.links[
                    x.points[0], v.points[0]]
                x.heap = [tup for tup in x.heap if tup[1] is not u and tup[1] is not v]
                g_measure = self.goodness_measure(w, x)
                x.heap.append((g_measure, w))
                heapq.heapify(x.heap)
                w.heap.append((g_measure, x))

            heapq.heapify(w.heap)
            self.global_heap.append(w)
            heapq.heapify(self.global_heap)
            print("Clusters left: {}".format(len(self.global_heap)))

    def compute_links(self) -> np.array:
        """Returns a matrix of links, O(n*m_a^2) complexity
        where m_a is an average number of neighbors for each cluster-point"""

        neighbors_list = self.find_neighbors()
        links = np.zeros((self.S.shape[0], self.S.shape[0]), dtype=int)
        n_rows, n_col = self.S.shape
        for i in range(0, n_rows):
            i_neighbors = neighbors_list[i][0]
            for j in range(0, i_neighbors.shape[0] - 1):
                for l in range(j + 1, i_neighbors.shape[0]):
                    links[i_neighbors[j], i_neighbors[l]] = links[i_neighbors[j], i_neighbors[l]] + 1
                    links[i_neighbors[l], i_neighbors[j]] = links[i_neighbors[l], i_neighbors[j]] + 1
            print("Links calcuated for {} points".format(i))
        return links

    def find_neighbors(self) -> list:
        n_rows, n_col = self.S.shape
        neighbors_list = [None for i in range(0, n_rows)]
        for i in range(0, n_rows):
            feature_similarities_and = np.logical_and(self.S, self.S[i, :])
            feature_similarities_and[i, :] = False  # So that point isn't his own neighbor
            feature_similarities_or = np.logical_or(self.S, self.S[i, :])
            similarity = tanimoto_coefficient(feature_similarities_and, feature_similarities_or)
            neighbors_list[i] = np.where(similarity >= self.nbr_threshold)
        return neighbors_list

    def goodness_measure(self, c1: Cluster, c2: Cluster) -> float:
        numerator = 0
        for i in c1.points:
            for j in c2.points:
                numerator += self.links[i, j]
        denominator = (len(c1.points) + len(c1.points)) ** self.goodness_exponent - len(
            c1.points) ** self.goodness_exponent - len(
            c2.points) ** self.goodness_exponent
        return (-1) * numerator / denominator

    def clusters(self) -> list:
        return [x for x in self.q if x is not None]
