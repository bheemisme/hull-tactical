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
import numpy as np


def predict(test: pl.DataFrame) -> float:
    """Replace this function with your inference code.
    You can return either a Pandas or Polars dataframe, though Polars is recommended for performance.
    Each batch of predictions (except the very first) must be returned within 5 minutes of the batch features being provided.
    """
    MIN_INVESTMENT = 0.0
    MAX_INVESTMENT = 2.0
    test = test.drop("date_id", "is_scored")

    test = pp.remove_dummy_features(test)

    model_fr = joblib.load("checkpoints/XGBRegressor_forward_returns_1.pkl")
    model_rfr = joblib.load("checkpoints/XGBRegressor_risk_free_rate_1.pkl")

  
    forward_return = model_fr.predict(test[0].to_numpy())[0]
    risk_free_rate = model_rfr.predict(test[0].to_numpy())[0]

    
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
    position = np.clip(position, MIN_INVESTMENT, MAX_INVESTMENT, dtype=np.float64)

   
    return position

   


# When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting
# or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very
# first `predict` call, which does not have the usual 1 minute response deadline.

inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(
    predict
)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(("data/",))
