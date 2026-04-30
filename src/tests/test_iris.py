import pandas as pd
import numpy as np
from src.modeling import ikmeans_initialize


def run_iris_validation():
    """
    Tests the iK-Means initialization procedure on the standard Iris dataset
    to validate anomalous cluster extraction logic against known benchmarks.
    """
    try:
        df = pd.read_csv("../data/iris.dat", header=None, sep=r"\s+")
        X_iris = df.iloc[:, :4].values.astype(np.float64)
    except Exception as e:
        print(f"Error loading 'iris.dat': {e}")
        return

    MIN_SIZE = 10
    UNIT_RANGES = True

    try:
        ap_clusters, _ = ikmeans_initialize(
            X=X_iris, min_cluster_size=MIN_SIZE, use_unit_ranges=UNIT_RANGES
        )

        retained_clusters = [ap for ap in ap_clusters if ap.size >= MIN_SIZE]

        print(f"Valid Clusters: {len(retained_clusters)} (Waiting: 3 Clusters)")
        print("-" * 50)

        for i, cluster in enumerate(retained_clusters):
            print(f"Cluster {i+1}:")
            print(f"  Size:        {cluster.size} points")
            print(f"  Real Centroid (Raw):  {np.round(cluster.centroid_raw, 2)}")
            print(f"  Contribution (Scatter): {cluster.scatter_pct:.1f}%")
            print("-" * 50)

    except Exception as e:
        print(f"Execution Error: {e}")


if __name__ == "__main__":
    run_iris_validation()
