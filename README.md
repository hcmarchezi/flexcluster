# flexcluster 

flexcluster is a python package that provides a flexible implementation for clustering algorithms based on K-means.

The package provides a generic clustering function that allows distance function customization with callback parameters:
* **dissimilarity function** - *function(datapoint1, datapoint2) : int* - function that defines the distance between 2 data points.
* **centroid calculation function** - *function(datapoints : np.array) : datapoint* - function that calculates a centroid given an array of datapoints.


Below is an example of a generic call to cluster elements based on custom dissimilarity and centroid calculation functions:

```python
from flexcluster import clustering

centroids, centroid_labels = clustering(
            data, # array of elements
            k=3,  # number of clusters
            dissimilarity_fn=dissimilarity_fn, # dissimilarity function
            centroid_calc_fn=centroid_calc_fn, # centroid calculation function
            max_tries=5) # number of clustering tries (get best result)
            
centroids # => calculated centroids per cluster
centroid_labels # => map with a numeric key for each cluster and value is an array of item indexes
```

Library already provides wrappers for most commonly used clustering strategies such as K-meand and K-medoids
which are provided as shown in the examples below:

```python
from flexcluster import kmeans

centroids, centroid_labels = kmeans(data, k=3)
```
```python
from flexcluster import kmedoids

centroids, centroid_labels = kmedoids(data, k=3)
```


