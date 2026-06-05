import os
import argparse
import pandas as pd
import datetime
from src.data_loader import load_data
from src.preprocessing import clean_and_engineer_data, build_preprocessor
from src.evaluation import evaluate_models

def main():
    parser = argparse.ArgumentParser(description="Run the unsupervised learning pipeline.")
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Run in fast mode with reduced K grid, fewer seeds, and a data subsample for quick testing."
    )
    args = parser.parse_args()

    initial = datetime.datetime.now()
    print(f"Starting pipeline (fast={args.fast})")

    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("Loading dataset and verifying hashes...")
    df = load_data("data/raw/hotel_bookings_course_release_v1.csv")

    # Fast mode configuration to save evaluator time
    sample_rule = "full_dataset"
    if args.fast:
        print("Fast mode activated: Subsampling data to 10,000 rows...")
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)
        sample_rule = "subsample_10k"
        k_range = [3, 4]
        seeds = [42]
    else:
        # Full grid as defined in the milestone report
        k_range = [3, 4, 5, 6, 7, 8, 11, 14] 
        seeds = [42, 123, 456, 789, 999]

    print("Extracting 'adr' and outcome variables for post-hoc profiling...")
    df_cleaned = clean_and_engineer_data(df)
    
    # save profiling base data
    profiling_data = df_cleaned[["is_canceled", "reservation_status", "adr"]].copy()
    profiling_data.to_csv("tables/profiling_base_data.csv", index=False)

    all_results = []

    # Iterate over scaler variants AND hotel inclusion to test feature dominance
    for scaler_type in ["standard", "robust"]:
        for include_hotel in [True, False]:
            hotel_tag = "withHotel" if include_hotel else "noHotel"
            rep_id = f"R-EUCLID-{scaler_type}-countryTop15-noADR-{hotel_tag}"
            print(f"Processing: {rep_id}")
            preprocessor = build_preprocessor(scaler_type=scaler_type, include_hotel=include_hotel)
            X_processed = preprocessor.fit_transform(df_cleaned)

            print(f"Matrix shape: {X_processed.shape[0]:,} rows and {X_processed.shape[1]} columns")
            print("Testing K-Means, GMM, and iK-Means...")

            results_df = evaluate_models(
                X_processed=X_processed,
                rep_id=rep_id,
                k_range=k_range,
                seeds=seeds,
                sample_rule=sample_rule,
            )
            all_results.extend(results_df)

    print("\nCompiling experiments.csv...")
    final_experiments_df = pd.DataFrame(all_results)
    final_experiments_df.to_csv("tables/experiments.csv", index=False)
    
    runtime = datetime.datetime.now() - initial
    print(f"\nPipeline finished successfully in {runtime}!")
    print("Results saved to tables/experiments.csv.")

if __name__ == "__main__":
    main()