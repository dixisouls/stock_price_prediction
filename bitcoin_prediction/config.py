"""
Configuration settings for the Bitcoin price prediction model.
"""

import os
import torch
from datetime import datetime
from typing import Dict, Any
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set random seeds for reproducibility
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    # Data settings
    "ticker": os.getenv("TICKER", "BTC-USD"),
    "start_date": os.getenv("START_DATE", "2018-01-01"),
    "end_date": os.getenv("END_DATE", "2024-12-31"),

    # Model settings
    "sequence_length": int(os.getenv("SEQUENCE_LENGTH", 60)),
    "d_model": int(os.getenv("D_MODEL", 512)),
    "nhead": int(os.getenv("NHEAD", 8)),
    "num_encoder_layers": int(os.getenv("NUM_ENCODER_LAYERS", 6)),
    "dim_feedforward": int(os.getenv("DIM_FEEDFORWARD", 2048)),
    "dropout": float(os.getenv("DROPOUT", 0.1)),

    # Training settings
    "train_split": float(os.getenv("TRAIN_SPLIT", 0.8)),
    "batch_size": int(os.getenv("BATCH_SIZE", 32)),
    "learning_rate": float(os.getenv("LEARNING_RATE", 0.0001)),
    "num_epochs": int(os.getenv("NUM_EPOCHS", 50)),

    # Prediction settings
    "num_predictions": int(os.getenv("NUM_PREDICTIONS", 30)),

    # Hardware settings
    "device": torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")),

    # Output settings
    "model_save_path": os.getenv("MODEL_SAVE_PATH", "models/bitcoin_transformer_model.pth"),
    "log_dir": os.getenv("LOG_DIR", "logs"),
    "output_dir": os.getenv("OUTPUT_DIR", "outputs"),
}


def get_config() -> Dict[str, Any]:
    """
    Get the configuration settings.

    Returns:
        Dictionary containing the configuration settings.
    """
    return DEFAULT_CONFIG