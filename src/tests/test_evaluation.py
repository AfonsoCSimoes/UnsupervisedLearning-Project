import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from src.evaluation import evaluate_models


def test_evaluation_loop():
    """
    Runs an isolated test for the evaluation script using mock data
    to verify that logging and clustering loops function correctly.
    """
    print("Running an isolated test for evalution.py")

    X_fake, _ = make_blobs(n_samples=500, n_features=10, centers=4, random_state=42)

    print("\nExecuting evaluation models...")
    df_results = evaluate_models(
        X_processed=X_fake, rep_id="TEST-REP", k_range=range(3, 5), seeds=[42, 99]
    )
    df_results_pd = pd.DataFrame(df_results)

    print("\nTest Evaluation - Done")
    print("First 3 log entries generated:")
    print(df_results_pd.head(3).to_string())


if __name__ == "__main__":
    test_evaluation_loop()
