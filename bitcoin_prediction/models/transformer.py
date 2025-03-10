"""
Transformer model implementation for the Bitcoin price prediction model.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from bitcoin_prediction.config import get_config


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer model.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional encoding.

        Args:
            d_model: Dimension of the model.
            max_len: Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0), :]
        return x


class CryptoTransformer(nn.Module):
    """
    Transformer model for cryptocurrency price prediction.
    """

    def __init__(
            self,
            feature_dim: int,
            d_model: Optional[int] = None,
            nhead: Optional[int] = None,
            num_encoder_layers: Optional[int] = None,
            dim_feedforward: Optional[int] = None,
            dropout: Optional[float] = None
    ):
        """
        Initialize the Transformer model.

        Args:
            feature_dim: Number of input/output features.
            d_model: Dimension of the model.
            nhead: Number of heads in the multihead attention.
            num_encoder_layers: Number of encoder layers.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout probability.
        """
        super(CryptoTransformer, self).__init__()

        # Get default values from config if not provided
        config = get_config()
        self.feature_dim = feature_dim
        self.d_model = d_model or config["d_model"]
        nhead = nhead or config["nhead"]
        num_encoder_layers = num_encoder_layers or config["num_encoder_layers"]
        dim_feedforward = dim_feedforward or config["dim_feedforward"]
        dropout = dropout or config["dropout"]

        # Input embedding
        self.embedding = nn.Linear(feature_dim, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            self.d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # Output layer
        self.output_layer = nn.Linear(self.d_model, feature_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: Input tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            Output tensor of shape (batch_size, feature_dim).
        """
        # Reshape input: (batch_size, seq_len, feature_dim) -> (seq_len, batch_size, feature_dim)
        src = src.permute(1, 0, 2)

        # Embedding
        src = self.embedding(src) * np.sqrt(self.d_model)

        # Positional encoding
        src = self.pos_encoder(src)

        # Transformer encoder
        output = self.transformer_encoder(src)

        # Use the last time step output for prediction
        output = output[-1]

        # Output layer
        output = self.output_layer(output)

        return output

    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path: Path to save the model.
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, feature_dim: int, device: torch.device = None) -> "CryptoTransformer":
        """
        Load the model from a file.

        Args:
            path: Path to load the model from.
            feature_dim: Number of input/output features.
            device: Device to load the model to.

        Returns:
            Loaded model.
        """
        model = cls(feature_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model