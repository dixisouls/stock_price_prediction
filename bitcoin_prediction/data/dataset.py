"""
Dataset class for the Bitcoin price prediction model.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

from bitcoin_prediction.config import get_config


class CryptoDataset(Dataset):
    """
    PyTorch Dataset for cryptocurrency price data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: Optional[int] = None,
        train: bool = True,
        train_split: Optional[float] = None,
        scale_data: bool = True,
        scaler: Optional[StandardScaler] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data: DataFrame containing price data with engineered features.
            sequence_length: Number of time steps to look back.
            train: Whether this is a training dataset or not.
            train_split: Fraction of data to use for training.
            scale_data: Whether to scale the data using StandardScaler.
            scaler: Scaler to use for scaling the data. If None, a new one is created.
        """
        # Get default values from config if not provided
        config = get_config()
        self.sequence_length = sequence_length or config["sequence_length"]
        train_split = train_split or config["train_split"]

        # Train/test split
        split_idx = int(len(data) * train_split)
        if train:
            self.data = data.iloc[:split_idx].values
        else:
            self.data = data.iloc[split_idx:].values

        # Store column names
        self.feature_columns = data.columns

        # Scale data
        self.scaler = scaler
        if scale_data:
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.data = self.scaler.fit_transform(self.data)
            else:
                self.data = self.scaler.transform(self.data)

        # Create sequences
        self.X, self.y = self._create_sequences()

    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences of data for the model.

        Returns:
            Tuple containing input sequences and target values.
        """
        X, y = [], []

        for i in range(len(self.data) - self.sequence_length):
            X.append(self.data[i : i + self.sequence_length])
            y.append(self.data[i + self.sequence_length])

        return np.array(X), np.array(y)

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple containing input sequence and target values.
        """
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return X, y

    @property
    def feature_dim(self) -> int:
        """
        Get the number of features.

        Returns:
            Number of features.
        """
        return self.data.shape[1]
