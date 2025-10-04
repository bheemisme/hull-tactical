"""
This module contains all functions for data visualization.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import eda.utils as ut
import eda.queries as queries
from typing import List


def plot_feature_target_scatter(train_df: pl.DataFrame, feature_type: ut.FeatureType, target_variables: List[str]):
    """
    Plot scatter plots for feature columns vs target variables using seaborn PairGrid.

    Parameters:
    train_df (pl.DataFrame): The training DataFrame.
    feature_type (str): The full name of the feature type (e.g., 'Macroeconomic').
    target_variables (list): List of target variable column names.

    Returns:
    None: Displays the plot.
    """
    # Get feature columns
    _, feature_columns = queries.get_feature_columns(train_df, feature_type)

    if not feature_columns:
        print(f"No columns found for feature type: {feature_type}")
        return

    # Convert to pandas for seaborn
    df_pandas = train_df.to_pandas()

    # Create PairGrid with y_vars as feature columns, x_vars as targets
    g = sns.PairGrid(df_pandas, x_vars=target_variables, y_vars=feature_columns, height=3, aspect=1)

    # Map scatter plots
    g.map(sns.scatterplot)

    # Set title
    g.figure.suptitle(f'Scatter Plots: {feature_type} Features vs Targets', y=1.02)

    # Show plot
    plt.show()

def plot_feature_variables(df: pl.DataFrame, feature_type: ut.FeatureType):
    """
    Plot scatter plots for each column of a feature type against date_id.

    Parameters:
    df (pl.DataFrame): The input DataFrame.
    feature_type (FeatureType): The feature type enum.

    Returns:
    None: Displays the plots.
    """
    # Get feature columns
    _, feature_columns = queries.get_feature_columns(df, feature_type)

    if not feature_columns:
        print(f"No columns found for feature type: {feature_type}")
        return

    # Number of columns
    n_cols = len(feature_columns)

    # Create subplots
    fig, axes = plt.subplots(nrows=n_cols, figsize=(10, 5 * n_cols))

    if n_cols == 1:
        axes.scatter(df['date_id'], df[feature_columns[0]])
        axes.set_title(f'{feature_columns[0]} over date_id')
        axes.set_xlabel('date_id')
        axes.set_ylabel(feature_columns[0])
    else:
        for ax, col in zip(axes, feature_columns):
            ax.scatter(df['date_id'], df[col])
            ax.set_title(f'{col} over date_id')
            ax.set_xlabel('date_id')
            ax.set_ylabel(col)

    plt.tight_layout()
    plt.show()

def plot_target_variables(df: pl.DataFrame):
    """
    Plot scatter plots for each target variable against date_id.

    Parameters:
    df (pl.DataFrame): The input DataFrame.

    Returns:
    None: Displays the plots.
    """
    target_variables = ut.TARGET_VARIABLES
    n_targets = len(target_variables)

    # Create subplots
    fig, axes = plt.subplots(nrows=n_targets, figsize=(10, 5 * n_targets))

    if n_targets == 1:
        axes.scatter(df['date_id'], df[target_variables[0]])
        axes.set_title(f'{target_variables[0]} over date_id')
        axes.set_xlabel('date_id')
        axes.set_ylabel(target_variables[0])
    else:
        for ax, target in zip(axes, target_variables):
            ax.scatter(df['date_id'], df[target])
            ax.set_title(f'{target} over date_id')
            ax.set_xlabel('date_id')
            ax.set_ylabel(target)

    plt.tight_layout()
    plt.show()