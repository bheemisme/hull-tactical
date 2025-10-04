import polars as pl
import eda.process as pp
import eda.utils as ut

from typing import Tuple

def preprocess(path: str) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load the training data from CSV file using Polars.

    Parameters:
    path (str): Path to the CSV file.

    Returns:
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]: X_train, X_eval, y_train, y_eval.
    """
    train_df = pl.read_csv(path)
    train_df = pp.remove_dummy_features(train_df)

    train_df = pp.cast_feature_type(train_df, ut.FeatureType.technical, pl.Float64)
    train_df = pp.cast_feature_type(train_df, ut.FeatureType.macroeconomic, pl.Float64)
    train_df = pp.cast_feature_type(train_df, ut.FeatureType.interest_rate, pl.Float64)
    train_df = pp.cast_feature_type(train_df, ut.FeatureType.price_valuation, pl.Float64)
    train_df = pp.cast_feature_type(train_df, ut.FeatureType.volatility, pl.Float64)
    train_df = pp.cast_feature_type(train_df, ut.FeatureType.sentiment, pl.Float64)
    
    train_df = pp.create_lagged_targets(train_df)
    train_df = train_df.fill_null(strategy='backward')

    eval_df = train_df[-180:]
    train_df = train_df[:-180]
    
    input_features = [col for col in train_df.columns if col not in ['date_id'] + ut.TARGET_VARIABLES]
    X_train = train_df[input_features]
    y_train = train_df[ut.TARGET_VARIABLES]
    
    X_eval = eval_df[input_features]
    y_eval = eval_df[ut.TARGET_VARIABLES]
    
    
    
    return X_train, X_eval, y_train, y_eval
