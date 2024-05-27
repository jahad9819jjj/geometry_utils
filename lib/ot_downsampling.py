import numpy as np

def sampling_farthest_points(points, colors=None, npoint=10000):
    """ Sampling by farthest.

    Args:
        points (np.array): NxD
        npoint (int): M

    Returns:
        np.array : MxD
    """
    np.random.seed(0)
    N, D = points.shape
    xyz = points[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    idxs = centroids.astype(np.int32)
    return idxs