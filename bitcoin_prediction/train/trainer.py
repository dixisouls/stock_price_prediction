"""
Training functions for the Bitcoin price prediction model.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from bitcoin_prediction.config import get_config
from bitcoin_prediction.utils.logger import logger


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        num_epochs: Optional[int] = None,
        device: Optional[torch.device] = None,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 10
) -> Dict[str, List[float]]:
    """
    Train the model.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        optimizer: Optimizer for training. If None, Adam with default learning rate is used.
        criterion: Loss function. If None, MSELoss is used.
        num_epochs: Number of training epochs. If None, use value from config.
        device: Device to use for training. If None, use value from config.
        save_path: Path to save the best model. If None, use value from config.
        early_stopping_patience: Number of epochs to wait for improvement before stopping.

    Returns:
        Dictionary containing training and validation losses.
    """
    # Get default values from config if not provided
    config = get_config()
    device = device or config["device"]
    num_epochs = num_epochs or config["num_epochs"]

    # Default optimizer and criterion
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    if criterion is None:
        criterion = nn.MSELoss()

    # Default save path
    if save_path is None:
        save_path = config["model_save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logger.info(f"Starting training on {device} for {num_epochs} epochs")

    model.to(device)
    history = {'train_loss': [], 'val_loss': []}

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.6f}, saving model to {save_path}")
            model.save(save_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(
                f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

    logger.info("Training complete")
    return history


def create_data_loaders(
        train_dataset,
        test_dataset,
        batch_size: Optional[int] = None,
        val_ratio: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        batch_size: Batch size. If None, use value from config.
        val_ratio: Fraction of training data to use for validation.

    Returns:
        Tuple containing training, validation, and test data loaders.
    """
    # Get default batch size from config if not provided
    config = get_config()
    batch_size = batch_size or config["batch_size"]

    # Calculate sizes for train and validation splits
    train_size = int((1 - val_ratio) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # Create train and validation datasets
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader