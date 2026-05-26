import time
import numpy as np
import pandas as pd
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.utils import resample
from sklearn.mixture import GaussianMixture
from src.modeling import ikmeans_initialize


def kmeans_bootstrap_stability(X, K, n_boot=20, seed=42):
    """
    Bootstrap stability for K-Means via pairwise ARI on overlapping records.

    For each pair of bootstrap samples, ARI is computed only on the intersection
    of record indices present in both samples. This avoids the need for label
    alignment across disjoint sets and is the same method used in notebook 03.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    n_samples = X.shape[0]
    boot_indices = []
    boot_labels = []

    for i in range(n_boot):
        indices = resample(np.arange(n_samples), random_state=seed+i)
        X_boot = X[indices]
        model = KMeans(n_clusters=K, random_state=seed+i, n_init=10)
        labels = model.fit_predict(X_boot)
        boot_indices.append(indices)
        boot_labels.append(labels)

    ari_scores = []
    for i, j in itertools.combinations(range(n_boot), 2):
        idx_i, idx_j = boot_indices[i], boot_indices[j]
        intersect = np.intersect1d(idx_i, idx_j)
        if len(intersect) == 0:
            continue

        def get_intersect_labels(idx_array, labels_array, intersect_pts):
            mapping = {pt: lbl for pt, lbl in zip(idx_array, labels_array)}
            return [mapping[pt] for pt in intersect_pts]

        labels_i = get_intersect_labels(idx_i, boot_labels[i], intersect)
        labels_j = get_intersect_labels(idx_j, boot_labels[j], intersect)
        ari_scores.append(adjusted_rand_score(labels_i, labels_j))

    if len(ari_scores) == 0:
        return {"mean_ARI": None, "std_ARI": None, "min_ARI": None, "max_ARI": None}

    return {
        "mean_ARI": np.mean(ari_scores),
        "std_ARI": np.std(ari_scores),
        "min_ARI": np.min(ari_scores),
        "max_ARI": np.max(ari_scores),
    }


def gmm_seed_stability(X, K, seeds, covariance_type="full", n_init=3):
    """
    Stability of GMM across seeds via pairwise ARI between hard assignments.
    Returns mean/std/min/max ARI across all seed pairs.
    """
    from sklearn.mixture import GaussianMixture

    all_labels = []
    for seed in seeds:
        gmm = GaussianMixture(
            n_components=K,
            covariance_type=covariance_type,
            random_state=seed,
            n_init=n_init,
        )
        gmm.fit(X)
        all_labels.append(gmm.predict(X))

    ari_scores = [
        adjusted_rand_score(all_labels[i], all_labels[j])
        for i, j in itertools.combinations(range(len(seeds)), 2)
    ]

    if len(ari_scores) == 0:
        return {"mean_ARI": None, "std_ARI": None, "min_ARI": None, "max_ARI": None}

    return {
        "mean_ARI": np.mean(ari_scores),
        "std_ARI":  np.std(ari_scores),
        "min_ARI":  np.min(ari_scores),
        "max_ARI":  np.max(ari_scores),
    }


def evaluate_models(
        X_processed, rep_id, k_range=range(3, 9), seeds=[42, 123, 456, 789, 999], sample_rule="full_dataset"):
    """
    Executes stability tests for standard K-Means across multiple Ks and seeds,
    initializes iK-Means to determine the optimal K, and generates the experiment logs
    tracking runtime, silhouette, calinski_harabasz, and davies_bouldin scores.
    """
    logs = []

    print("\nStarting Standard K-Means evaluation")
    for k in k_range:
        stability_metrics = kmeans_bootstrap_stability(
            X_processed, k, n_boot=20, seed=42)
        for seed in seeds:
            print(f"Training K-Means: K={k:02d} | Seed={seed}")

            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)

            start_time = time.time()
            labels = kmeans.fit_predict(X_processed)
            runtime = time.time() - start_time

            logs.append(
                {
                    "representation_id": rep_id,
                    "method": "k-means",
                    "k": k,
                    "seed": seed,
                    "silhouette": silhouette_score(
                        X_processed, labels, sample_size=30000
                    ),
                    "calinski_harabasz": calinski_harabasz_score(X_processed, labels),
                    "davies_bouldin": davies_bouldin_score(X_processed, labels),
                    "runtime_seconds": runtime,
                    "sample_rule": sample_rule,
                    "parameters": "n_init=10",
                    "mean_ARI": stability_metrics["mean_ARI"],
                    "std_ARI": stability_metrics["std_ARI"],
                    "min_ARI": stability_metrics["min_ARI"],
                    "max_ARI": stability_metrics["max_ARI"],
                }
            )

    print("\nInitializing GMM")
    for k in k_range:
        gmm_stability = gmm_seed_stability(X_processed, k, seeds=seeds)
        for seed in seeds:
            print(f"Training GMM: K={k:02d} | Seed={seed}")

            start_time = time.time()

            gmm = GaussianMixture(
                n_components=k, covariance_type="full", random_state=seed, n_init=3)
            gmm.fit(X_processed)
            labels_gmm = gmm.predict(X_processed)

            runtime = time.time() - start_time

            logs.append(
                {
                    "representation_id": rep_id,
                    "method": "gmm",
                    "k": k,
                    "seed": seed,
                    "silhouette": silhouette_score(
                        X_processed, labels_gmm, sample_size=30000
                    ),
                    "calinski_harabasz": calinski_harabasz_score(X_processed, labels_gmm),
                    "davies_bouldin": davies_bouldin_score(X_processed, labels_gmm),
                    "runtime_seconds": runtime,
                    "sample_rule": sample_rule,
                    "parameters": "covariance_type=full",
                    "mean_ARI": gmm_stability["mean_ARI"],
                    "std_ARI": gmm_stability["std_ARI"],
                    "min_ARI": gmm_stability["min_ARI"],
                    "max_ARI": gmm_stability["max_ARI"],
                }
            )

    print("\nInitializing iK-MEANS")
    try:
        start_time_ik = time.time()

        min_size = 100
        _, init_centroids = ikmeans_initialize(
            X=X_processed, min_cluster_size=min_size, use_unit_ranges=True
        )

        k_ik = len(init_centroids)
        print(f"Success! iK-Means determined K={k_ik} clusters.")

        if k_ik >= 2:
            kmeans_ik = KMeans(n_clusters=k_ik, init=init_centroids, n_init=1)
            labels_ik = kmeans_ik.fit_predict(X_processed)
            runtime_ik = time.time() - start_time_ik

            logs.append(
                {
                    "representation_id": rep_id,
                    "method": "ik-means",
                    "k": k_ik,
                    "seed": "deterministic",
                    "silhouette": silhouette_score(
                        X_processed, labels_ik, sample_size=30000
                    ),
                    "calinski_harabasz": calinski_harabasz_score(
                        X_processed, labels_ik
                    ),
                    "davies_bouldin": davies_bouldin_score(X_processed, labels_ik),
                    "runtime_seconds": runtime_ik,
                    "sample_rule": sample_rule,
                    "parameters": f"min_cluster_size={min_size}",
                    "mean_ARI": None,
                    "std_ARI": None,
                    "min_ARI": None,
                    "max_ARI": None
                }
            )
    except Exception as e:
        print(f"iK-Means failed: {e}")

    return logs
