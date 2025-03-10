# Bitcoin Price Prediction with PyTorch Transformer

A production-ready implementation of a Transformer model for predicting Bitcoin prices and other market variables.

## Project Overview

This project implements a PyTorch Transformer architecture for time series forecasting, specifically designed to predict Bitcoin (BTC) prices. Unlike traditional prediction models that only focus on closing prices, this implementation "extends the table" by predicting all features simultaneously, creating a more comprehensive view of future market conditions.

## Features

- Fetches historical Bitcoin data using `yfinance`
- Implements extensive feature engineering with technical indicators
- Uses PyTorch Transformer architecture for time series prediction
- Predicts not only closing prices but all engineered features
- Includes comprehensive evaluation metrics and visualization tools
- Implements early stopping and model checkpointing
- Follows production-ready Python project structure

## Technical Indicators

The model uses a comprehensive set of technical indicators:

- Simple Moving Averages (SMA-7, SMA-14, SMA-21)
- Exponential Moving Averages (EMA-9, EMA-21)
- Relative Strength Index (RSI-14)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Average True Range (ATR-14)
- Volume-based indicators
- Price Momentum
- Lagged features

## Project Structure

```
bitcoin_prediction/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── bitcoin_prediction/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── acquisition.py
│   │   ├── dataset.py
│   │   └── features.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   └── visualization.py
│   └── train/
│       ├── __init__.py
│       ├── trainer.py
│       └── evaluator.py
└── scripts/
    └── train_model.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dixisouls/stock_price_prediction.git
cd stock_price_prediction
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Usage

### Basic Usage

To train the model with default parameters:

```bash
python scripts/train_model.py
```

### Advanced Usage

The script supports many command-line options:

```bash
python scripts/train_model.py --ticker BTC-USD --start-date 2020-01-01 --num-epochs 100 --learning-rate 0.0001
```

### Prediction Only

To load a trained model and make predictions without training:

```bash
python scripts/train_model.py --load-model --no-train
```

## Configuration

The project uses a configuration system with defaults that can be overridden by environment variables or command-line arguments. See `config.py` for available options.

You can also create a `.env` file in the project root to set environment variables:

```
TICKER=BTC-USD
START_DATE=2019-01-01
END_DATE=2024-12-31
NUM_EPOCHS=75
LEARNING_RATE=0.0002
```

## Output

The model generates several outputs:

- Trained model weights
- Prediction CSV files
- Visualization plots
  - Actual vs Predicted prices
  - Training history
  - Feature importance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Divya Panchal