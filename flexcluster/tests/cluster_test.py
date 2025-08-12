import numpy as np
from flexcluster import clustering
from flexcluster import kmeans
from flexcluster import kmedoids


class TestKmedoids:
    def test_clustering(self):
        data = [1, 104, 51, 105, 4, 53, 9, 103, 52]
        initial_centroids = [20, 70, 100]

        centroids, centroid_labels = kmedoids(
            data,
            k=3,
            stop_criteria=0.1,
            initial_centroids=initial_centroids,
            max_tries=1)

        assert {4, 52, 104} == set(centroids)
        assert {0, 4, 6} == set(centroid_labels[0])
        assert {2, 5, 8} == set(centroid_labels[1])
        assert {1, 3, 7} == set(centroid_labels[2])


class TestKmeans:
    def test_clustering(self):
        data = [3, 104, 51, 105, 4, 53, 8, 103, 52]
        initial_centroids = [20, 70, 100]

        centroids, centroid_labels = kmeans(
            data,
            k=3,
            stop_criteria=0.1,
            initial_centroids=initial_centroids,
            max_tries=1)

        assert {5, 52, 104} == set(centroids)
        assert {0, 4, 6} == set(centroid_labels[0])
        assert {2, 5, 8} == set(centroid_labels[1])
        assert {1, 3, 7} == set(centroid_labels[2])


class TestClustering:
    def test_clustering(self):
        data = [2, 104, 51, 105, 4, 53, 3, 103, 52]
        def dissimilarity_fn(item1, item2): return np.abs(item2 - item1)
        def centroid_calc_fn(arr): return np.mean(arr)
        initial_centroids = [20, 70, 100]

        centroids, centroid_labels = clustering(
            data,
            k=3,
            dissimilarity_fn=dissimilarity_fn,
            centroid_calc_fn=centroid_calc_fn,
            stop_criteria=0.1,
            initial_centroids=initial_centroids,
            max_tries=1)

        assert {3, 52, 104} == set(centroids)
        assert {0, 4, 6} == set(centroid_labels[0])
        assert {2, 5, 8} == set(centroid_labels[1])
        assert {1, 3, 7} == set(centroid_labels[2])
