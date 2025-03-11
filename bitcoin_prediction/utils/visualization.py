"""
Visualization functions for the Bitcoin price prediction model.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from bitcoin_prediction.config import get_config
from bitcoin_prediction.utils.logger import logger


def plot_predictions(
    actual: pd.DataFrame,
    predictions: pd.DataFrame,
    feature: str = "Close",
    output_dir: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """
    Plot actual vs predicted values for a specific feature.

    Args:
        actual: DataFrame containing actual values.
        predictions: DataFrame containing predicted values.
        feature: Feature to plot.
        output_dir: Directory to save the plot. If None, use the default.
        show_plot: Whether to display the plot.

    Returns:
        Path to the saved plot.
    """
    plt.figure(figsize=(14, 7))

    plt.plot(actual.index, actual[feature], label="Actual")
    plt.plot(predictions.index, predictions[feature], label="Predicted", linestyle="--")

    plt.title(f"{feature} Price - Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # Create output directory if it doesn't exist
    if output_dir is None:
        config = get_config()
        output_dir = config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_path = os.path.join(output_dir, f"{feature}_prediction.png")
    plt.savefig(output_path)

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    logger.info(f"Plot saved as {output_path}")

    return output_path


def plot_training_history(
    history: Dict[str, List[float]],
    output_dir: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """
    Plot training and validation loss.

    Args:
        history: Dictionary containing training and validation losses.
        output_dir: Directory to save the plot. If None, use the default.
        show_plot: Whether to display the plot.

    Returns:
        Path to the saved plot.
    """
    plt.figure(figsize=(14, 7))

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Create output directory if it doesn't exist
    if output_dir is None:
        config = get_config()
        output_dir = config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(output_path)

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    logger.info(f"Training history plot saved as {output_path}")

    return output_path


def plot_feature_importance(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = "MSE",
    top_n: int = 10,
    output_dir: Optional[str] = None,
    show_plot: bool = False,
) -> str:
    """
    Plot feature importance based on prediction metrics.

    Args:
        metrics: Dictionary containing evaluation metrics for each feature.
        metric_name: Name of the metric to use for importance.
        top_n: Number of top features to show.
        output_dir: Directory to save the plot. If None, use the default.
        show_plot: Whether to display the plot.

    Returns:
        Path to the saved plot.
    """
    # Extract metric values for each feature
    features = []
    values = []

    for feature, feature_metrics in metrics.items():
        features.append(feature)
        values.append(feature_metrics[metric_name])

    # Sort by metric value
    sorted_indices = np.argsort(values)

    if metric_name in ["MSE", "MAE", "RMSE"]:
        # For error metrics, lower is better
        sorted_indices = sorted_indices[:top_n]
    else:
        # For other metrics, higher is better
        sorted_indices = sorted_indices[-top_n:]

    top_features = [features[i] for i in sorted_indices]
    top_values = [values[i] for i in sorted_indices]

    plt.figure(figsize=(14, 10))

    plt.barh(top_features, top_values)

    plt.title(f"Top {top_n} Features by {metric_name}")
    plt.xlabel(metric_name)
    plt.ylabel("Feature")
    plt.grid(True, axis="x")

    # Create output directory if it doesn't exist
    if output_dir is None:
        config = get_config()
        output_dir = config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_path = os.path.join(output_dir, f"feature_importance_{metric_name}.png")
    plt.savefig(output_path)

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    logger.info(f"Feature importance plot saved as {output_path}")

    return output_path
