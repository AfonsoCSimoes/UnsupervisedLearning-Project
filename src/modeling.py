import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.cluster import KMeans

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class APCluster:
    indices: list[int]
    centroid_raw: FloatArray
    centroid_std: FloatArray
    size: int
    scatter_pct: float


def compute_feature_statistics(
    X: FloatArray, use_unit_ranges: bool = False
) -> Tuple[FloatArray, FloatArray, float]:
    """
    Computes the grand mean, feature scales, and total normalized scatter.
    """
    mu = np.mean(X, axis=0)

    if use_unit_ranges:
        r = np.ones_like(mu)
    else:
        r = np.ptp(X, axis=0)
        r[r == 0] = 1.0

    Y = (X - mu) / r
    D = np.sum(Y**2)

    return mu, r, D


def normalized_squared_distances(
    X: FloatArray, indices: list[int], scales: FloatArray, reference: FloatArray
) -> FloatArray:
    """
    Returns the normalized squared distances of selected rows to a reference point.
    """
    X_sub = X[indices]
    return np.sum(((X_sub - reference) / scales) ** 2, axis=1)


def cluster_centroid(X: FloatArray, indices: list[int]) -> FloatArray:
    """
    Returns the component-wise mean of the selected rows.
    """
    return np.mean(X[indices], axis=0)


def extract_anomalous_cluster(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    mean: FloatArray,
    initial_centroid: FloatArray,
    seed_index: int,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> Tuple[list[int], FloatArray]:
    """
    Alternates assignment and centroid update to extract one anomalous cluster.
    """
    c = initial_centroid.copy()
    S_prev = []

    for _ in range(max_iter):
        dist_c = normalized_squared_distances(X, indices, scales, c)
        dist_mu = normalized_squared_distances(X, indices, scales, mean)

        mask = dist_c < dist_mu
        S = [indices[idx] for idx, val in enumerate(mask) if val]

        if not S:
            S = [seed_index]

        c_new = cluster_centroid(X, S)

        if S == S_prev:
            return S, c_new

        if np.linalg.norm(c_new - c) <= tol:
            return S, c_new

        c = c_new
        S_prev = S

    return S, c


def ikmeans_initialize(
    X: FloatArray,
    min_cluster_size: int,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    use_unit_ranges: bool = False,
) -> Tuple[List[APCluster], FloatArray]:
    """
    Main iK-means initialization procedure. Identifies anomalous clusters
    and returns them along with their standardized initial centroids.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    mu, r, D = compute_feature_statistics(X, use_unit_ranges)

    remains = list(range(n))
    ap_clusters = []

    while remains:
        dist_mu = normalized_squared_distances(X, remains, r, mu)
        q_local = np.argmax(dist_mu)
        q = remains[q_local]
        seed = X[q].copy()

        S, c = extract_anomalous_cluster(X, remains, r, mu, seed, q, tol, max_iter)
        z = (c - mu) / r

        if D > 0:
            scatter_pct = 100 * len(S) * np.sum(z**2) / D
        else:
            scatter_pct = 0.0

        ap_clusters.append(
            APCluster(
                indices=sorted(S),
                centroid_raw=c,
                centroid_std=z,
                size=len(S),
                scatter_pct=scatter_pct,
            )
        )

        remains = [i for i in remains if i not in S]

    retained = [ap for ap in ap_clusters if ap.size >= min_cluster_size]

    if not retained:
        raise ValueError("No anomalous cluster satisfies the minimum size.")

    init_centroids = np.vstack([ap.centroid_std for ap in retained])

    return ap_clusters, init_centroids
