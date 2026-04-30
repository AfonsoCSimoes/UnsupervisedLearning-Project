import os
import pandas as pd
import datetime
from src.data_loader import load_data
from src.preprocessing import clean_and_engineer_data, build_preprocessor
from src.evaluation import evaluate_models


def main():
    """
    Main execution pipeline. Sets up directories, loads and cleans data,
    extracts profiling variables, applies preprocessing, evaluates clustering
    models (Standard and Robust variants), and saves the results to experiments.csv.
    """
    initial = datetime.datetime.now()
    print("Starting Unsupervised Learning Pipeline...")

    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("Loading dataset and verifying hashes...")
    df = load_data("data/raw/hotel_bookings_course_release_v1.csv")

    print("Extracting 'adr' and outcome variables for post-hoc profiling...")
    df_cleaned = clean_and_engineer_data(df)
    profiling_data = df_cleaned[["is_canceled", "reservation_status", "adr"]].copy()
    profiling_data.to_csv("tables/profiling_base_data.csv", index=False)

    all_results = []

    for scaler_type in ["standard", "robust"]:
        rep_id = f"R-EUCLID-{scaler_type}-countryTop15-noADR"
        preprocessor = build_preprocessor(scaler_type=scaler_type)
        X_processed = preprocessor.fit_transform(df_cleaned)

        print(
            f"Matrix shape: {X_processed.shape[0]:,} rows and {X_processed.shape[1]} columns"
        )
        print("Testing K-Means and iK-Means...")

        results_df = evaluate_models(X_processed=X_processed, rep_id=rep_id)
        all_results.extend(results_df)

    print("\nCompiling experiments.csv...")
    final_experiments_df = pd.DataFrame(all_results)
    final_experiments_df.to_csv("tables/experiments.csv", index=False)

    final_time = datetime.datetime.now()
    print(f"\nPipeline completed successfully in {final_time - initial}.")
    print("Check the 'tables' folder for profiling data and experiments.csv.")


if __name__ == "__main__":
    main()
