import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer


def clean_and_engineer_data(df):
    """
    Performs initial pandas-level cleaning and feature engineering
    before passing to the scikit-learn pipeline.
    """

    df = df.copy()

    # Impute the 4 missing 'children' values with 0
    df["children"] = df["children"].fillna(0)

    # Derive requested behavioral variables
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["party_size"] = df["adults"] + df["children"] + df["babies"]

    # Group rare countries (Top 15 + 'Other')
    top_15_countries = df["country"].value_counts().nlargest(15).index
    df["country_grouped"] = df["country"].apply(
        lambda x: x if x in top_15_countries else "Other"
    )

    # Fill 'Unknown' for categorical missingness (just in case)
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


def build_preprocessor():
    """
    Builds the scikit-learn ColumnTransformer to create the R-EUCLID matrix.
    """

    # Variables that need log1p transformation to handle heavy right-skew
    log_num_features = [
        "lead_time",
        "previous_cancellations",
        "previous_bookings_not_canceled",
    ]

    # Standard numerical variables
    standard_num_features = ["total_nights", "party_size"]

    # Categorical variables for Full One-Hot Encoding
    cat_features = [
        "distribution_channel",
        "market_segment",
        "deposit_type",
        "hotel",
        "customer_type",
        "country_grouped",
    ]

    # Pipeline for log-transformed numericals
    log_num_pipeline = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline for standard numericals
    standard_num_pipeline = Pipeline([("scaler", StandardScaler())])

    # Pipeline for categoricals (No scaling applied to 0/1 columns as per rules)
    cat_pipeline = Pipeline(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("log_num", log_num_pipeline, log_num_features),
            ("std_num", standard_num_pipeline, standard_num_features),
            ("cat", cat_pipeline, cat_features),
        ],
        remainder="drop",  # Drops any columns not explicitly listed above (Leakage control!)
    )

    return preprocessor
