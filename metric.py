import numpy as np
import pandas as pd
import pandas.api.types
import polars as pl
import preprocess as pp
import joblib

MIN_INVESTMENT = 0
MAX_INVESTMENT = 2


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).

    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.

    Returns:
        float: The calculated adjusted Sharpe ratio.
    """

    # if not pandas.api.types.is_numeric_dtype(submission['prediction']):
    #     raise ParticipantVisibleError('Predictions must be numeric')

    # solution = solution
    # solution['position'] = submission['prediction']

    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].min()} below minimum of {MIN_INVESTMENT}')

    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr
    )
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)

def compute_position_single_trade(forward_return, risk_free_rate):
    """
    Compute the position for a single trade using only the trade's forward return and risk-free rate.
    
    Args:
        forward_return (float): forward return of the risky asset
        risk_free_rate (float): risk-free return for the same period
    
    Returns:
        float: position (between MIN_INVESTMENT and MAX_INVESTMENT)
    """
    # Step 1: compute excess return
    excess_return = forward_return - risk_free_rate

    # Step 2: allocate proportionally to the sign of excess return
    # Positive excess return → invest more; negative → invest minimum (0)
    if excess_return <= 0:
        position = MIN_INVESTMENT
    else:
        # Scale position by magnitude of excess return, but clip to max
        # Here we choose a simple linear scaling: 1% excess → 1 unit position
        # You can tune the scaling factor for risk appetite
        position = min(MAX_INVESTMENT, excess_return / 0.01)

    # Step 3: clip to bounds (redundant but safe)
    position = np.clip(position, MIN_INVESTMENT, MAX_INVESTMENT)

    return position

def predict():
    
    X_train, X_eval, _, _ = pp.preprocess('data/train.csv')
    
    model_fr = joblib.load("checkpoints/XGBRegressor_forward_returns_1.pkl")
    model_rfr = joblib.load("checkpoints/XGBRegressor_risk_free_rate_1.pkl")
    
    X_train_pd = X_train.to_pandas()
    X_eval_pd = X_eval.to_pandas()

    X_train =  X_train.with_columns([
        pl.Series('forward_returns', model_fr.predict(X_train_pd)),
        pl.Series('risk_free_rate', model_rfr.predict(X_train_pd))
    ])
    
    X_train = X_train.with_columns(
        pl.struct(['forward_returns', 'risk_free_rate']).map_elements(lambda s: compute_position_single_trade(s['forward_returns'], s['risk_free_rate']), return_dtype=pl.Float64).alias("position")
    )
    
    X_train = X_train.with_columns(pl.lit(0.5).alias('position'))
    X_eval =  X_eval.with_columns([
        pl.Series('forward_returns', model_fr.predict(X_eval_pd)),
        pl.Series('risk_free_rate', model_rfr.predict(X_eval_pd))
    ])
    X_eval = X_eval.with_columns(
        pl.struct(['forward_returns', 'risk_free_rate']).map_elements(lambda s: compute_position_single_trade(s['forward_returns'], s['risk_free_rate']), return_dtype=pl.Float64).alias("position")
    )
    
    X_eval = X_eval.to_pandas()
    X_train = X_train.to_pandas()
    
    train_score = score(X_train)
    eval_score = score(X_eval)
    
    print(f'train score: {train_score}, eval score: {eval_score}')
    


    
if __name__ == '__main__':
    predict()