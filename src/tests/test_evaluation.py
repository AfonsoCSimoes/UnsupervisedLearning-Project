import numpy as np
from sklearn.datasets import make_blobs
from evaluation import evaluate_models
import os


def test_evaluation_loop():
    print("Running a isolated test for evalution.py")

    # Generate fake data with 500 points, 10 variables, in 4 clusters.
    X_fake, _ = make_blobs(n_samples=500, n_features=10,
                           centers=4, random_state=42)

    # Run our evaluate_models function.
    # Using a small sample (K=3 and K=4) and only 2 seeds.
    print("\nRunning our evaluation.py")
    df_results = evaluate_models(
        X_processed=X_fake, k_range=range(3, 5), seeds=[42, 99])

    print("\nTest Evaluation - Done")
    print("First 3 lines that were generated.")
    print(df_results.head(3).to_string())


if __name__ == "__main__":
    test_evaluation_loop()
