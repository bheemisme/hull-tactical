"""
The evaluation API requires that you set up a server which will respond to inference requests.
We have already defined the server; you just need write the predict function.
When we evaluate your submission on the hidden test set the client defined in `default_gateway` will run in a different container
with direct access to the hidden test set and hand off the data timestep by timestep.

Your code will always have access to the published copies of the copmetition files.
"""

import os

import pandas as pd
import polars as pl

import kaggle_evaluation.default_inference_server
import eda.process as pp
import joblib

def predict(test: pl.DataFrame) -> float:
    """Replace this function with your inference code.
    You can return either a Pandas or Polars dataframe, though Polars is recommended for performance.
    Each batch of predictions (except the very first) must be returned within 5 minutes of the batch features being provided.
    """
    
    test = test.drop('date_id', 'is_scored')
    
    test = pp.remove_dummy_features(test)
    
    model_fr = joblib.load("checkpoints/XGBRegressor_forward_test_1.pkl")
    model_rfr = joblib.load("checkpoints/XGBRegressor_risk_free_rate_1.pkl")
    
    fr =model_fr.predict(test[0].to_numpy())
    rfr = model_rfr.predict(test[0].to_numpy())
    
    
    # remove dummy features.
    return 1


# When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting
# or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very
# first `predict` call, which does not have the usual 1 minute response deadline.

inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('data/',))
    