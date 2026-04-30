import pandas as pd
import hashlib
import os


def load_data(file_path="data/raw/hotel_bookings_course_release_v1.csv"):
    """
    Loads the dataset from the specified path and verifies its integrity.
    Computes and prints the SHA-256 hash of the raw data for governance.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Please download it."
        )

    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"Data Governance - SHA-256 Hash of raw data: {file_hash}")

    df = pd.read_csv(file_path)
    return df
