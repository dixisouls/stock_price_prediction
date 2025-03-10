"""
Training and evaluation modules for the Bitcoin price prediction model.
"""

from bitcoin_prediction.train.trainer import train_model
from bitcoin_prediction.train.evaluator import evaluate_model, make_predictions

__all__ = ["train_model", "evaluate_model", "make_predictions"]