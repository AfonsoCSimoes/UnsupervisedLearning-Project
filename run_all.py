import os
import pandas as pd

# These will be the modules you build in the src/ folder
# from src.data_loader import load_data
# from src.preprocessing import build_preprocessor
# from src.modeling import run_baselines
# from src.evaluation import log_experiments


def main():
    print("Starting Unsupervised Learning Pipeline...")

    # 1. Load Data
    # print("Loading dataset and verifying hashes...")
    # df = load_data('data/raw/hotel_bookings_course_release_v1.csv')

    # 2. Preprocessing
    # print("Building preprocessing pipeline...")
    # preprocessor = build_preprocessor()
    # X_processed = preprocessor.fit_transform(df)

    # 3. Modeling
    # print("Running Baseline Models (K-means, iK-means)...")
    # models = run_baselines(X_processed, k_range=range(3, 9), seeds=5)

    # 4. Evaluation & Logging
    # print("Computing internal indices and logging to experiments.csv...")
    # log_experiments(models, X_processed, file_path='experiments.csv')

    print("Pipeline complete. Check experiments.csv for results.")


if __name__ == "__main__":
    main()
