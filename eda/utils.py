"""
This module contains general utility functions.
"""

from enum import Enum


class FeatureType(Enum):
    technical = "Technical/Market Dynamics"
    macroeconomic = "Macroeconomic"
    interest_rate = "Interest Rate"
    price_valuation = "Price/Valuation"
    volatility = "Volatility"
    sentiment = "Sentiment"
    momentum = "Momentum"
    dummy = "Dummy/Binary"


# Dictionaries for feature type mappings
PREFIX_TO_TYPE = {
    "M": FeatureType.technical,
    "E": FeatureType.macroeconomic,
    "I": FeatureType.interest_rate,
    "P": FeatureType.price_valuation,
    "V": FeatureType.volatility,
    "S": FeatureType.sentiment,
    "MOM": FeatureType.momentum,
    "D": FeatureType.dummy,
}

TYPE_TO_PREFIX = {v: k for k, v in PREFIX_TO_TYPE.items()}

TARGET_VARIABLES = [
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
]

