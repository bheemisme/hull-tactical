import pandas as pd
import numpy as np
import eda.utils as ut
import joblib
import polars as pl

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from preprocess import preprocess
from pprint import pprint

# Load data
train_df = pl.read_csv('data/train.csv')
X_train, X_eval, y_train, y_eval = preprocess('data/train.csv')

X_train = X_train.to_pandas()
X_eval = X_eval.to_pandas()
y_train = y_train.to_pandas()
y_eval = y_eval.to_pandas()


# Define models
models = {
    'XGBRegressor': XGBRegressor(random_state=42),
    'LGBMRegressor': LGBMRegressor(random_state=42),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42)
}

def train(idx, results_df):
    # Evaluate each model for each target
    results = {
        "idx": [],
        "model": [],
        "target": [],
        "train_mse": [],
        "eval_mse": []
    }
    for target in ut.TARGET_VARIABLES:

        
        y_train_target = y_train[target]
        y_eval_target = y_eval[target]

        print(f"\nEvaluating models for target: {target}")
        for name, model in models.items():

            # Fit
            model.fit(X_train, y_train_target)

            # Predict on train and eval
            y_train_pred = model.predict(X_train)
            y_eval_pred = model.predict(X_eval)

            # Compute MSE
            train_mse = mean_squared_error(y_train_target, y_train_pred)
            eval_mse = mean_squared_error(y_eval_target, y_eval_pred)

            results['idx'].append(idx)
            results['model'].append(name)
            results['target'].append(target)
            results['train_mse'].append(train_mse)
            results['eval_mse'].append(eval_mse)

            print(f"  {name} - Train MSE: {train_mse:.4f}, Eval MSE: {eval_mse:.4f}")
            
            model_path = f'./checkpoints/{name}_{target}_{idx}.pkl'
            joblib.dump(model, model_path)
            print(f'model saved to {model_path}')


    new_results_df = pd.DataFrame(results)
    if results_df is not None:
        results_df = pd.concat((results_df, new_results_df), axis=0)
    else:
        results_df = new_results_df
        
    results_df.to_csv("./results.csv", index=False)
    pprint(results_df)

    print("\nModel evaluation completed for all targets.")
    return results_df

if __name__ == '__main__':
    # results_df = pd.read_csv('./results.csv')
    results_df = train(idx=1, results_df=None)
    

