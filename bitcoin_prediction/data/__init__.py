"""
Data processing modules for the Bitcoin price prediction model.
"""

from bitcoin_prediction.data.acquisition import fetch_crypto_data
from bitcoin_prediction.data.features import engineer_features
from bitcoin_prediction.data.dataset import CryptoDataset

__all__ = ["fetch_crypto_data", "engineer_features", "CryptoDataset"]
