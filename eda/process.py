"""
This module contains all functions for processing the DataFrame.
"""
import polars as pl
import eda.utils as ut

def remove_dummy_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove all dummy features (columns starting with 'D') from the DataFrame.

    Parameters:
    df (pl.DataFrame): The input DataFrame.

    Returns:
    pl.DataFrame: The DataFrame with dummy features removed.
    """
    dummy_cols = [col for col in df.columns if col.startswith("D")]
    df_cleaned = df.drop(dummy_cols)
    return df_cleaned


def cast_feature_type(df: pl.DataFrame, feature_type: ut.FeatureType, target_type: pl.DataType) -> pl.DataFrame:
    """
    Cast all columns of a specific feature type to a target data type.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    feature_prefix (str): The prefix of the feature type (e.g., 'E' for Macroeconomic).
    target_type (pl.DataType): The target Polars data type to cast to (e.g., pl.Float64).

    Returns:
    pl.DataFrame: The DataFrame with the specified feature type columns cast to the target type.
    """
    feature_prefix = ut.TYPE_TO_PREFIX[feature_type]
    matching_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    df_casted = df.with_columns(
        [pl.col(col).cast(target_type) for col in matching_cols]
    )
    return df_casted

def create_lagged_targets(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create lagged variables for each target variable by shifting 1 step and filling nulls with zero.

    Parameters:
    df (pl.DataFrame): The input DataFrame.

    Returns:
    pl.DataFrame: The DataFrame with added lagged target columns.
    """
    lagged_cols = []
    for target in ut.TARGET_VARIABLES:
        lagged_col_name = f"lagged_{target}"
        lagged_cols.append(pl.col(target).shift(1).fill_null(0).alias(lagged_col_name))

    df_with_lags = df.with_columns(lagged_cols)
    return df_with_lags
