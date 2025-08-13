from typing import Callable, Optional, Tuple, List, Dict, Any
import numpy as np
from flexcluster.impl.cluster_impl import _clustering


def kmedoids(
    data: List,
    k: int,
    stop_criteria: float = 0.1,
    initial_centroids: Optional[List] = None,
    max_tries: int = 5
) -> Tuple[List, Dict[int, List[int]]]:
    def dissimilarity_fn(item1, item2): return np.abs(item2 - item1)
    def centroid_calc_fn(arr): return np.median(arr)
    return _clustering(
        data, k, dissimilarity_fn, centroid_calc_fn, stop_criteria, initial_centroids, max_tries
    )


def kmeans(
    data: List,
    k: int,
    stop_criteria: float = 0.1,
    initial_centroids: Optional[List] = None,
    max_tries: int = 5
) -> Tuple[List, Dict[int, List[int]]]:
    def dissimilarity_fn(item1, item2): return np.abs(item2 - item1)
    def centroid_calc_fn(arr): return np.mean(arr)
    return _clustering(
        data, k, dissimilarity_fn, centroid_calc_fn, stop_criteria, initial_centroids, max_tries
    )


def clustering(
    data: List,
    k: int,
    dissimilarity_fn: Callable[[Any, Any], float],
    centroid_calc_fn: Callable[[List], Any],
    stop_criteria: float = 0.1,
    initial_centroids: Optional[List] = None,
    max_tries: int = 5
) -> Tuple[List, Dict[int, List[int]]]:
    return _clustering(
        data, k, dissimilarity_fn, centroid_calc_fn, stop_criteria, initial_centroids, max_tries
    )
