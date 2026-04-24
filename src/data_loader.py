import pandas as pd
import hashlib
import os


def load_data(file_path="data/raw/hotel_bookings_course_release_v1.csv"):
    """Loads the dataset and verifies its existence."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. Please download it."
        )
    df = pd.read_csv(file_path)
    return df
