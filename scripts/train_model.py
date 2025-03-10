"""
Main script to run the Bitcoin price prediction pipeline.
"""

import os
import argparse

import torch.nn as nn
import torch.optim as optim

from bitcoin_prediction.config import get_config
from bitcoin_prediction.data import fetch_crypto_data, engineer_features, CryptoDataset
from bitcoin_prediction.models import CryptoTransformer
from bitcoin_prediction.train import train_model, evaluate_model, make_predictions
from bitcoin_prediction.train.trainer import create_data_loaders
from bitcoin_prediction.utils import logger
from bitcoin_prediction.utils.visualization import plot_predictions, plot_training_history, plot_feature_importance


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Bitcoin Price Prediction")

    # Data arguments
    parser.add_argument("--ticker", type=str, help="Ticker symbol")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")

    # Model arguments
    parser.add_argument("--sequence-length", type=int, help="Sequence length")
    parser.add_argument("--d-model", type=int, help="Model dimension")
    parser.add_argument("--nhead", type=int, help="Number of heads")
    parser.add_argument("--num-encoder-layers", type=int, help="Number of encoder layers")
    parser.add_argument("--dim-feedforward", type=int, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, help="Dropout probability")

    # Training arguments
    parser.add_argument("--train-split", type=float, help="Train split ratio")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, help="Number of epochs")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")

    # Prediction arguments
    parser.add_argument("--num-predictions", type=int, help="Number of predictions")

    # Output arguments
    parser.add_argument("--model-save-path", type=str, help="Model save path")
    parser.add_argument("--no-train", action="store_true", help="Skip training")
    parser.add_argument("--load-model", action="store_true", help="Load model for prediction")

    return parser.parse_args()


def update_config_from_args(args):
    """
    Update config with command line arguments.

    Args:
        args: Command line arguments.

    Returns:
        Updated config dictionary.
    """
    config = get_config()

    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None:
            # Convert hyphens to underscores in key
            key = key.replace("-", "_")
            config[key] = value

    return config


def main():
    """
    Main function to run the Bitcoin price prediction pipeline.
    """
    # Parse command line arguments
    args = parse_args()

    # Update config with command line arguments
    config = update_config_from_args(args)

    # Create output directories
    os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    logger.info("Starting Bitcoin price prediction pipeline")
    logger.info(f"Using device: {config['device']}")

    # Fetch data
    data = fetch_crypto_data(
        ticker=config["ticker"],
        start_date=config["start_date"],
        end_date=config["end_date"]
    )

    # Engineer features
    data_with_features = engineer_features(data)

    # Create datasets
    train_dataset = CryptoDataset(
        data_with_features,
        sequence_length=config["sequence_length"],
        train=True,
        train_split=config["train_split"]
    )

    test_dataset = CryptoDataset(
        data_with_features,
        sequence_length=config["sequence_length"],
        train=False,
        train_split=config["train_split"],
        scale_data=True,
        scaler=train_dataset.scaler
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        test_dataset,
        batch_size=config["batch_size"]
    )

    # Get feature dimension
    feature_dim = train_dataset.feature_dim

    # Check if we need to load an existing model
    if args.load_model:
        logger.info(f"Loading model from {config['model_save_path']}")
        model = CryptoTransformer.load(
            config["model_save_path"], feature_dim, device=config["device"]
        )
    else:
        # Create model
        model = CryptoTransformer(
            feature_dim=feature_dim,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"]
        )

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()

    # Train model if needed
    if not args.no_train and not args.load_model:
        logger.info("Training model")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config["num_epochs"],
            device=config["device"],
            save_path=config["model_save_path"],
            early_stopping_patience=args.early_stopping_patience
        )

        # Plot training history
        plot_training_history(history, output_dir=config["output_dir"])

    # Load the best model
    model = CryptoTransformer.load(
        config["model_save_path"], feature_dim, device=config["device"]
    )

    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        scaler=train_dataset.scaler,
        feature_columns=data_with_features.columns,
        device=config["device"]
    )

    # Plot feature importance
    plot_feature_importance(metrics, output_dir=config["output_dir"])

    # Make future predictions
    predictions = make_predictions(
        model=model,
        data=data_with_features,
        scaler=train_dataset.scaler,
        sequence_length=config["sequence_length"],
        num_predictions=config["num_predictions"],
        device=config["device"]
    )

    # Plot results
    plot_predictions(
        actual=data_with_features.iloc[-90:],
        predictions=predictions,
        feature='Close',
        output_dir=config["output_dir"]
    )

    # Save predictions
    predictions_path = os.path.join(config["output_dir"], "predictions.csv")
    predictions.to_csv(predictions_path)
    logger.info(f"Predictions saved to {predictions_path}")

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()