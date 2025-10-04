"""
This module contains all functions for querying the DataFrame.
"""

from typing import List, Tuple
import polars as pl
import eda.utils as ut



def get_feature_type_samples(df: pl.DataFrame, n_samples=10, seed=42) -> pl.DataFrame:
    """
    Return a DataFrame with random non-null samples from columns of all feature types.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    n_samples (int): Number of random samples to fetch per feature type (default 10).
    seed (int): Random seed for reproducibility (default 42).

    Returns:
    pl.DataFrame: A DataFrame with 'Feature Type' and 'Sample Values' columns for each feature type.
    """
    results = []
    for feature_type in ut.TYPE_TO_PREFIX.keys():
        feature_prefix = ut.TYPE_TO_PREFIX[feature_type]
        matching_cols = [col for col in df.columns if col.startswith(feature_prefix)]

        all_samples = []
        for col in matching_cols:
            non_null_values = df[col].drop_nulls()
            if len(non_null_values) > 0:
                sample_size = min(n_samples, len(non_null_values))
                sampled = non_null_values.sample(sample_size, seed=seed)
                all_samples.extend(sampled.to_list())

        # If more than n_samples total, sample again from all_samples
        if len(all_samples) > n_samples:
            import random

            random.seed(seed)
            all_samples = random.sample(all_samples, n_samples)

        results.append({"Feature Type": feature_type.value, "Sample Values": all_samples})

    result_df = pl.DataFrame(results)
    return result_df


def get_feature_columns(df: pl.DataFrame, feature_type: ut.FeatureType) -> Tuple[int, List[str]]:
    """
    Get the total count and list of columns for a specific feature type.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    feature_type (str): The full name of the feature type (e.g., 'Macroeconomic').

    Returns:
    tuple: (total_columns_count, list_of_columns)
    """
    if feature_type not in ut.TYPE_TO_PREFIX:
        raise ValueError(f"Unknown feature type: {feature_type}")

    feature_prefix = ut.TYPE_TO_PREFIX[feature_type]
    matching_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    total_count = len(matching_cols)

    return total_count, matching_cols


def get_dataframe_info(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate a summary DataFrame with information about each column in the input DataFrame.

    Parameters:
    df (pl.DataFrame): The Polars DataFrame to analyze.

    Returns:
    pl.DataFrame: A DataFrame containing column names, data types, null counts, and total row count.
    """
    columns = df.columns
    dtypes = [str(df.schema[col]) for col in columns]
    null_counts = [df[col].null_count() for col in columns]
    total_count = df.height

    info_df = pl.DataFrame(
        {
            "Column": columns,
            "Data Type": dtypes,
            "Null Count": null_counts,
            "Total Count": [total_count] * len(columns),
        }
    )

    return info_df


def get_feature_type_column_counts(df: pl.DataFrame) -> pl.DataFrame:
    """
    Count the number of columns for each feature type based on column prefixes and list unique data types.

    Parameters:
    df (pl.DataFrame): The Polars DataFrame to analyze.

    Returns:
    pl.DataFrame: A DataFrame with feature types, their column counts, and unique data types.
    """
    results = []
    for prefix, name in ut.PREFIX_TO_TYPE.items():
        matching_cols = [col for col in df.columns if col.startswith(prefix)]
        count = len(matching_cols)
        unique_dtypes = set(str(df.schema[col]) for col in matching_cols)
        unique_dtypes_str = ", ".join(sorted(unique_dtypes))

        results.append(
            {
                "Feature Type": name.value,
                "Column Count": count,
                "Unique Data Types": unique_dtypes_str,
            }
        )

    counts_df = pl.DataFrame(results)

    return counts_df

def get_feature_type_info(df: pl.DataFrame, feature_type: ut.FeatureType) -> pl.DataFrame:
    """
    Get statistical information for columns of a specific feature type.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    feature_type (FeatureType): The feature type enum.

    Returns:
    pl.DataFrame: DataFrame with columns: Column, Null Count, Non-Null Count, Mean, Std Dev, Skewness, Kurtosis.
    """
    if feature_type not in ut.TYPE_TO_PREFIX:
        raise ValueError(f"Unknown feature type: {feature_type}")

    feature_prefix = ut.TYPE_TO_PREFIX[feature_type]
    matching_cols = [col for col in df.columns if col.startswith(feature_prefix)]

    results = []
    for col in matching_cols:
        null_count = df[col].null_count()
        total_count = df.height
        non_null_count = total_count - null_count

        # Compute statistics (assuming numerical columns)
        mean_val = df[col].mean()
        std_val = df[col].std()
        skew_val = df[col].skew()
        kurt_val = df[col].kurtosis()

        results.append({
            'Column': col,
            'Null Count': null_count,
            'Non-Null Count': non_null_count,
            'Mean': mean_val,
            'Std Dev': std_val,
            'Skewness': skew_val,
            'Kurtosis': kurt_val
        })

    return pl.DataFrame(results)

def get_feature_counts(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get null and non-null counts for each feature type by summing counts across associated columns.

    Parameters:
    df (pl.DataFrame): The input DataFrame.

    Returns:
    pl.DataFrame: DataFrame with columns: feature_type, Null count, Non Null count.
    """
    results = []
    for feature_type in ut.TYPE_TO_PREFIX.keys():
        feature_prefix = ut.TYPE_TO_PREFIX[feature_type]
        matching_cols = [col for col in df.columns if col.startswith(feature_prefix)]

        total_null_count = 0
        total_non_null_count = 0

        for col in matching_cols:
            null_count = df[col].null_count()
            total_count = df.height
            non_null_count = total_count - null_count

            total_null_count += null_count
            total_non_null_count += non_null_count

        results.append({
            'feature_type': feature_type.value,
            'Null count': total_null_count,
            'Non Null count': total_non_null_count
        })

    return pl.DataFrame(results)
