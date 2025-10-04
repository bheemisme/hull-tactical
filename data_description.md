# Data Description

Data is historic market data. The coverage stretches back decades; expect to see extensive missing values early on. Data is int the format of a csv file. In the data, the each row gives information about trade that happened on a particular day.

## Train Data

- file path: data/train.csv

### Index Variable

- date_id is the index variable, representing day on which the trade happened

### Target Variables

The Target variables in the data with their corresponding data type:

1. forward_returns
The returns from buying the S&P 500 and selling it a day later.

2. risk_free_rate
The federal funds rate. Train set only.

3. market_forward_excess_returns
Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4.

All three target variables have zero null values. These variables are only found in training set, these are the variables to be predicted for.

### Input Feature Variables

The Input feature variables are categoriesed into following

1. All variables starts with 'M' are Market Dynamics or Technical features.
2. All variables starts with 'E' are Macro Economic features.
3. All variables starts with 'I' are Interest Rate features.
4. All variables starts with 'P' are Price/Valuation features.
5. All variables starts with 'V' are Volatility features.
6. All variables starts with 'S' are Sentiment features.
7. All variables starts with 'MOM' are Momentum features.
8. All variables starts with 'D' are Dummy or Binary features.

## Test Data

This is a mock test set representing the structure of the unseen test set, which will be used in testing phase.

- file path: data/test.csv

### Test Set Input Feature Variables

The input feature variables used in test set are

1. Includes all feature variables in train.csv.

2. **lagged_forward_returns** - The returns from buying the S&P 500 and selling it a day later, provided with a lag of one day.

3. **lagged_risk_free_rate** - The federal funds rate, provided with a lag of one day.

4. **lagged_market_forward_excess_returns** - Forward returns relative to expectations. Computed by subtracting the rolling five-year mean forward returns and winsorizing the result using a median absolute deviation (MAD) with a criterion of 4, provided with a lag of one day.
