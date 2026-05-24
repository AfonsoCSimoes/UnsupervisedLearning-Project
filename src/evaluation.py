import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from src.modeling import ikmeans_initialize


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
                    "parameters": "n_init=10"
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
                    "parameters": f"min_cluster_size={min_size}"
                }
            )
    except Exception as e:
        print(f"iK-Means failed: {e}")

    return logs
