from typing import Callable, Optional, Tuple, List, Dict, Any
import numpy as np


def _clustering(
    data: List,
    k: int,
    dissimilarity_fn: Callable[[Any, Any], float],
    centroid_calc_fn: Callable[[List], Any],
    stop_criteria: float = 0.1,
    initial_centroids: Optional[List] = None,
    max_tries: int = 5
) -> Tuple[List, Dict[int, List[int]]]:
    if initial_centroids is None:
        centroids = _choose_initial_centroids(data, k)
    else:
        centroids = initial_centroids

    best_result_cluster_cost = 10000000
    best_result_centroids = None
    best_result_centroid_labels = None

    while max_tries > 0:
        centroid_labels = None
        diff = 100000

        centroid_labels, centroids = _clustering_algorithm(
            centroid_calc_fn, centroid_labels, centroids, data, diff, dissimilarity_fn, stop_criteria
        )

        cluster_cost = _calculate_cluster_cost(data, dissimilarity_fn, centroids, centroid_labels)

        if cluster_cost < best_result_cluster_cost:
            best_result_centroids = centroids
            best_result_centroid_labels = centroid_labels

        max_tries -= 1

    return best_result_centroids, best_result_centroid_labels


def _clustering_algorithm(
    centroid_calc_fn: Callable[[List], Any],
    centroid_labels: Optional[Dict[int, List[int]]],
    centroids: List,
    data: List,
    diff: float,
    dissimilarity_fn: Callable[[Any, Any], float],
    stop_criteria: float
) -> Tuple[Dict[int, List[int]], List]:
    while diff > stop_criteria:
        centroid_labels = _find_nearest_centroid(data, centroids, dissimilarity_fn)
        new_centroids = _calculate_new_centroids(data, centroid_labels, centroid_calc_fn, centroids)
        diff = _average_centroids_move(centroids, new_centroids)
        centroids = new_centroids
    return centroid_labels, centroids


def _calculate_cluster_cost(
    data: List,
    dissimilarity_fn: Callable[[Any, Any], float],
    centroids: List,
    centroid_labels: Dict[int, List[int]]
) -> float:
    distance = 0
    for centroid_index in range(len(centroids)):
        centroid = centroids[centroid_index]
        for item_index in centroid_labels[centroid_index]:
            item = data[item_index]
            distance += dissimilarity_fn(centroid, item)

    return distance / len(data)


def _choose_initial_centroids(data: List, k: int) -> List:
    centroid_idxs = np.random.randint(data.shape[0], size=k)
    return data[centroid_idxs]


def _find_nearest_centroid(
    data: List,
    centroids: List,
    dissimilarity: Callable[[Any, Any], float]
) -> Dict[int, List[int]]:
    centroid_labels = {}
    for idx in range(len(centroids)):
        centroid_labels[idx] = []

    for item_idx, item in enumerate(data):
        min_dist = None
        centroid_id = None
        for centroid_idx, c in enumerate(centroids):
            distance = dissimilarity(item, c)
            if min_dist is None or distance < min_dist:
                min_dist = distance
                centroid_id = centroid_idx
        centroid_labels[centroid_id].append(item_idx)

    return centroid_labels


def _calculate_new_centroids(
    data: List,
    centroid_labels: Dict[int, List[int]],
    centroid_calc_fn: Callable[[List], Any],
    original_centroids: List
) -> List:
    centroids = []

    for centroid_idx in range(len(original_centroids)):
        indices = centroid_labels[centroid_idx]
        cluster_data = [item for idx, item in enumerate(data) if idx in indices]
        original_centroid = original_centroids[centroid_idx]
        new_centroid = _calculate_new_centroid(
            cluster_data,
            centroid_calc_fn=centroid_calc_fn,
            original_centroid=original_centroid
        )
        centroids.append(new_centroid)

    return np.array(centroids)


def _calculate_new_centroid(
    cluster_data: List,
    centroid_calc_fn: Callable[[List], Any],
    original_centroid
):
    if len(cluster_data) == 0:
        return original_centroid
    else:
        return centroid_calc_fn(cluster_data)


def _average_centroids_move(
    centroids: List,
    new_centroids: List,
    dissimilarity_fn: Optional[Callable[[Any, Any], float]] = None
) -> float:
    if dissimilarity_fn is None:
        def dissimilarity_fn(a, b): return np.abs(a - b)
    result = []
    for index in range(len(centroids)):
        result.append(dissimilarity_fn(centroids[index], new_centroids[index]))
    return np.mean(result)
