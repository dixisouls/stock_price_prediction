"""
Visualization functions for the Bitcoin price prediction model.
"""

import os
from pathlib import Path
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

    # Plot actual values
    plt.plot(actual.index, actual[feature], label="Actual", color="blue")

    # Plot predicted values with a solid line but different color
    plt.plot(
        predictions.index,
        predictions[feature],
        label="Predicted",
        color="orange",
        linestyle="-",
    )

    # To create a continuous visual, add a subtle connection between the last actual and first predicted point
    plt.plot(
        [actual.index[-1], predictions.index[0]],
        [actual[feature].iloc[-1], predictions[feature].iloc[0]],
        color="gray",
        alpha=0.5,
        linestyle="-",
        linewidth=1,
    )

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

    # Convert to lists to ensure hashable types
    features = [str(f) for f in features]
    values = [float(v) for v in values]

    # Create tuples of (feature, value) for sorting
    feature_value_pairs = list(zip(features, values))

    # Sort by metric value
    if metric_name in ["MSE", "MAE", "RMSE"]:
        # For error metrics, lower is better
        feature_value_pairs.sort(key=lambda x: x[1])
        # Take top_n
        feature_value_pairs = feature_value_pairs[:top_n]
    else:
        # For other metrics, higher is better
        feature_value_pairs.sort(key=lambda x: x[1], reverse=True)
        # Take top_n
        feature_value_pairs = feature_value_pairs[:top_n]

    # Unzip the pairs
    top_features, top_values = (
        map(list, zip(*feature_value_pairs)) if feature_value_pairs else ([], [])
    )

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
