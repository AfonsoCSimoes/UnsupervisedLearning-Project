import pandas as pd
import hashlib
import os


def load_data(file_path="data/raw/hotel_bookings_course_release_v1.csv"):
    """Loads the dataset and verifies its existence and cryptographic hash."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Please download it."
        )

    # Compute and print file hash for data governance
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Data Governance - SHA-256 Hash of raw data: {file_hash}")

    df = pd.read_csv(file_path)
    return df
