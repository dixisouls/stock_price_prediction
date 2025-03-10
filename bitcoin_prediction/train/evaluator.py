"""
Evaluation functions for the Bitcoin price prediction model.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Optional

from bitcoin_prediction.config import get_config
from bitcoin_prediction.utils.logger import logger


def evaluate_model(
        model: nn.Module,
        test_loader: DataLoader,
        scaler: StandardScaler,
        feature_columns: List[str],
        device: Optional[torch.device] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model on the test set.

    Args:
        model: The trained model.
        test_loader: DataLoader for the test data.
        scaler: Scaler used to normalize the data.
        feature_columns: List of feature column names.
        device: Device to use for evaluation. If None, use value from config.

    Returns:
        Dictionary containing evaluation metrics for each feature.
    """
    # Get default device from config if not provided
    if device is None:
        config = get_config()
        device = config["device"]

    logger.info(f"Evaluating model on test set using {device}")

    model.to(device)
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)

            outputs = model(batch_X)

            all_targets.append(batch_y.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)

    # Inverse transform to get actual values
    if scaler is not None:
        all_targets = scaler.inverse_transform(all_targets)
        all_predictions = scaler.inverse_transform(all_predictions)

    # Calculate metrics for each feature
    metrics = {}

    for i, feature in enumerate(feature_columns):
        target = all_targets[:, i]
        pred = all_predictions[:, i]

        mse = mean_squared_error(target, pred)
        mae = mean_absolute_error(target, pred)

        metrics[feature] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': np.sqrt(mse)
        }

        logger.info(f"Metrics for {feature}: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={np.sqrt(mse):.6f}")

    logger.info("Evaluation complete")
    return metrics


def make_predictions(
        model: nn.Module,
        data: pd.DataFrame,
        scaler: StandardScaler,
        sequence_length: Optional[int] = None,
        num_predictions: Optional[int] = None,
        device: Optional[torch.device] = None
) -> pd.DataFrame:
    """
    Make future predictions using the trained model.

    Args:
        model: The trained model.
        data: DataFrame containing the latest data.
        scaler: Scaler used to normalize the data.
        sequence_length: Number of time steps to look back. If None, use value from config.
        num_predictions: Number of future time steps to predict. If None, use value from config.
        device: Device to use for prediction. If None, use value from config.

    Returns:
        DataFrame containing the predictions.
    """
    # Get default values from config if not provided
    config = get_config()
    sequence_length = sequence_length or config["sequence_length"]
    num_predictions = num_predictions or config["num_predictions"]
    device = device or config["device"]

    logger.info(f"Making {num_predictions} future predictions using {device}")

    model.to(device)
    model.eval()

    # Prepare the latest sequence
    latest_data = data.values[-sequence_length:]

    if scaler is not None:
        latest_data = scaler.transform(latest_data)

    # Make predictions
    predictions = []

    # Initial sequence
    current_sequence = latest_data.copy()

    for _ in range(num_predictions):
        # Convert to tensor
        tensor_input = torch.tensor(current_sequence.reshape(1, sequence_length, -1), dtype=torch.float32).to(device)

        # Get prediction
        with torch.no_grad():
            next_pred = model(tensor_input).cpu().numpy()[0]

        predictions.append(next_pred)

        # Update sequence for the next prediction
        current_sequence = np.vstack([current_sequence[1:], next_pred])

    # Inverse transform predictions
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)

    # Create DataFrame for predictions
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=num_predictions, freq='D')
    predictions_df = pd.DataFrame(predictions, index=future_dates, columns=data.columns)

    logger.info("Predictions complete")
    return predictions_df