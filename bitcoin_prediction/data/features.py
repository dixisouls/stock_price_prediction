"""
Feature engineering functions for the Bitcoin price prediction model.
"""

import pandas as pd
from typing import List, Tuple

from bitcoin_prediction.utils.logger import logger


def calculate_sma(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average for a given column.

    Args:
        data: DataFrame containing price data.
        column: Column name to calculate SMA for.
        window: Window size for the moving average.

    Returns:
        Series containing the SMA values.
    """
    return data[column].rolling(window=window).mean()


def calculate_ema(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average for a given column.

    Args:
        data: DataFrame containing price data.
        column: Column name to calculate EMA for.
        window: Window size for the moving average.

    Returns:
        Series containing the EMA values.
    """
    return data[column].ewm(span=window, adjust=False).mean()


def calculate_rsi(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    """
    Calculate Relative Strength Index for a given column.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss

    Args:
        data: DataFrame containing price data.
        column: Column name to calculate RSI for.
        window: Window size for the RSI.

    Returns:
        Series containing the RSI values.
    """
    delta = data[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(
    data: pd.DataFrame,
    column: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence) for a given column.

    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA of MACD Line
    Histogram = MACD Line - Signal Line

    Args:
        data: DataFrame containing price data.
        column: Column name to calculate MACD for.
        fast_period: Period for the fast EMA.
        slow_period: Period for the slow EMA.
        signal_period: Period for the signal line.

    Returns:
        Tuple containing the MACD line, signal line, and histogram.
    """
    fast_ema = calculate_ema(data, column, fast_period)
    slow_ema = calculate_ema(data, column, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    data: pd.DataFrame, column: str, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands for a given column.

    Middle Band = SMA
    Upper Band = Middle Band + (Standard Deviation * num_std)
    Lower Band = Middle Band - (Standard Deviation * num_std)

    Args:
        data: DataFrame containing price data.
        column: Column name to calculate Bollinger Bands for.
        window: Window size for the moving average.
        num_std: Number of standard deviations for the bands.

    Returns:
        Tuple containing the upper band, middle band, and lower band.
    """
    middle_band = calculate_sma(data, column, window)
    std_dev = data[column].rolling(window=window).std()

    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)

    return upper_band, middle_band, lower_band


def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = Moving average of True Range

    Args:
        data: DataFrame containing price data with High, Low, and Close columns.
        window: Window size for the moving average.

    Returns:
        Series containing the ATR values.
    """
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()

    return atr


def calculate_volume_sma(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average for volume.

    Args:
        data: DataFrame containing price data with Volume column.
        window: Window size for the moving average.

    Returns:
        Series containing the Volume SMA values.
    """
    return data["Volume"].rolling(window=window).mean()


def add_lagged_features(
    data: pd.DataFrame, columns: List[str], lags: List[int]
) -> pd.DataFrame:
    """
    Add lagged features for the specified columns.

    Args:
        data: DataFrame containing price data.
        columns: List of column names to create lagged features for.
        lags: List of lag periods.

    Returns:
        DataFrame with lagged features added.
    """
    df = data.copy()

    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the raw price data.

    Args:
        data: DataFrame containing raw price data.

    Returns:
        DataFrame with engineered features.
    """
    logger.info("Engineering features")

    df = data.copy()

    # Calculate SMAs
    df["SMA_7"] = calculate_sma(df, "Close", 7)
    df["SMA_14"] = calculate_sma(df, "Close", 14)
    df["SMA_21"] = calculate_sma(df, "Close", 21)

    # Calculate EMAs
    df["EMA_9"] = calculate_ema(df, "Close", 9)
    df["EMA_21"] = calculate_ema(df, "Close", 21)

    # Calculate RSI
    df["RSI_14"] = calculate_rsi(df, "Close", 14)

    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(df, "Close")
    df["MACD_Line"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Histogram"] = histogram

    # Calculate Bollinger Bands
    upper_band, middle_band, lower_band = calculate_bollinger_bands(df, "Close")
    df["BB_Upper"] = upper_band
    df["BB_Middle"] = middle_band
    df["BB_Lower"] = lower_band

    # Calculate Average True Range
    df["ATR_14"] = calculate_atr(df, 14)

    # Calculate Volume SMA
    df["Volume_SMA_7"] = calculate_volume_sma(df, 7)
    df["Volume_SMA_14"] = calculate_volume_sma(df, 14)

    # Add price momentum
    df["Price_Momentum"] = df["Close"] / df["Close"].shift(7) - 1

    # Add lagged features
    lag_columns = ["Close", "SMA_7", "RSI_14", "MACD_Line"]
    df = add_lagged_features(df, lag_columns, [1, 2, 3])

    # Drop rows with NaN values
    df = df.dropna()

    logger.info(f"Feature engineering complete. Shape: {df.shape}")

    return df
