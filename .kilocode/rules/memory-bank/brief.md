# Brief

## Project Overview

Hull Tactical Market Prediction - Kaggle competition project focused on building machine learning models to predict three key financial metrics: forward_returns, risk_free_rate, and market_forward_excess_returns for optimal portfolio positioning.

## Core Problem

Develop ML models that can predict market returns and optimal position sizing for a tactical investment strategy. The models must handle multiple feature categories and be evaluated using a custom volatility-adjusted Sharpe ratio metric.

## Key Components

- **Data Processing**: Handle historic market data spanning decades with extensive early missing values, formatted as CSV with daily trade rows indexed by date_id
- **Feature Engineering**: Process 100+ input features across 8 categories:
  - Technical/Market Dynamics (M): M1-M18
  - Macroeconomic (E): E1-E20
  - Interest Rate (I): I1-I9
  - Price/Valuation (P): P1-P13
  - Volatility (V): V1-V13
  - Sentiment (S): S1-S12
  - Momentum (MOM): M1-M18
  - Dummy/Binary (D): D1-D9
- **Model Development**: Build predictive models for three target variables:
  - forward_returns: Daily S&P 500 returns (buy and sell next day)
  - risk_free_rate: Federal funds rate (train set only)
  - market_forward_excess_returns: Excess returns vs. 5-year rolling mean (winsorized)
- **Risk Management**: Implement position sizing constraints (0-2 range) with volatility penalties exceeding 1.2x market volatility
- **Evaluation**: Custom metric calculation using volatility-adjusted Sharpe ratio with return and volatility penalties

## Technical Scope

- Multi-modal feature processing pipeline
- Time series prediction models
- Custom evaluation framework
- Position optimization within risk constraints
