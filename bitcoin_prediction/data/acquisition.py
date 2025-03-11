"""
Data acquisition functions for the Bitcoin price prediction model.
"""

import pandas as pd
import yfinance as yf
from typing import Optional

from bitcoin_prediction.config import get_config
from bitcoin_prediction.utils.logger import logger


def fetch_crypto_data(
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical cryptocurrency data using yfinance.

    Args:
        ticker: The ticker symbol for the cryptocurrency.
        start_date: The start date for the data in YYYY-MM-DD format.
        end_date: The end date for the data in YYYY-MM-DD format.

    Returns:
        DataFrame containing the historical data.
    """
    # Use default values from config if not provided
    config = get_config()
    ticker = ticker or config["ticker"]
    start_date = start_date or config["start_date"]
    end_date = end_date or config["end_date"]

    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")

    try:
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        logger.info(f"Successfully fetched {len(data)} records")
        return data

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise
