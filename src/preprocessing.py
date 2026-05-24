import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)


def clean_and_engineer_data(df):
    """
    Performs initial pandas-level cleaning and feature engineering.
    Imputes missing 'children' values with 0, derives 'total_nights' and 'party_size',
    groups rare countries keeping only the Top 15 (rest as 'Other'),
    and fills missing categorical values with 'Unknown'.
    """
    df = df.copy()

    df = df[(df['adr'] >= 0) & (df['adr'] <= 5000)]

    df["children"] = df["children"].fillna(0)
    df["total_nights"] = df["stays_in_weekend_nights"] + \
        df["stays_in_week_nights"]
    df["party_size"] = df["adults"] + df["children"] + df["babies"]

    top_15_countries = df["country"].value_counts().nlargest(15).index
    df["country_grouped"] = df["country"].apply(
        lambda x: x if x in top_15_countries else "Other"
    )

    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df


def build_preprocessor(scaler_type="standard", include_hotel=True):
    """
    Builds the scikit-learn ColumnTransformer to create the R-EUCLID matrix.
    Applies log1p transformation to heavy right-skewed numericals,
    scales numericals using either StandardScaler or RobustScaler,
    and applies full One-Hot Encoding to categoricals without scaling the 0/1 outputs.
    Drops any unlisted columns to control data leakage.
    """
    log_num_features = [
        "lead_time",
        "previous_cancellations",
        "previous_bookings_not_canceled",
    ]
    standard_num_features = ["total_nights", "party_size"]

    cat_features = [
        "distribution_channel",
        "market_segment",
        "deposit_type",
        "customer_type",
        "country_grouped",
        "arrival_date_month",
    ]

    if include_hotel:
        cat_features.append("hotel")

    chosen_scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()

    log_num_pipeline = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", chosen_scaler),
        ]
    )

    standard_num_pipeline = Pipeline([("scaler", chosen_scaler)])

    cat_pipeline = Pipeline(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("log_num", log_num_pipeline, log_num_features),
            ("std_num", standard_num_pipeline, standard_num_features),
            ("cat", cat_pipeline, cat_features),
        ],
        remainder="drop",
    )

    return preprocessor
