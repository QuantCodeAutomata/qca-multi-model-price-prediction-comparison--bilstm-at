"""
Data acquisition module for Gold (GC=F) and Bitcoin (BTC-USD) daily closing prices.

Downloads from Yahoo Finance via yfinance. Gold data is weekday-only with forward-fill
for missing values. Bitcoin data uses all calendar days.
"""

import os
import pandas as pd
import yfinance as yf


START_DATE = "2015-01-01"
END_DATE = "2021-09-11"  # exclusive end date for yfinance (downloads through 2021-09-10)


def download_gold(start: str = START_DATE, end: str = END_DATE) -> pd.Series:
    """
    Download Gold spot daily closing prices from Yahoo Finance (ticker GC=F).

    Parameters
    ----------
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date (exclusive) in YYYY-MM-DD format.

    Returns
    -------
    pd.Series
        Gold daily closing prices indexed by date, weekday-only, NaNs forward-filled.
    """
    df = yf.download("GC=F", start=start, end=end, progress=False, auto_adjust=False)
    close = df["Close"].squeeze().dropna()
    # Forward-fill any remaining missing weekday entries
    close = close.ffill()
    close.name = "Gold_Close"
    return close


def download_bitcoin(start: str = START_DATE, end: str = END_DATE) -> pd.Series:
    """
    Download Bitcoin daily closing prices from Yahoo Finance (ticker BTC-USD).

    Parameters
    ----------
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date (exclusive) in YYYY-MM-DD format.

    Returns
    -------
    pd.Series
        Bitcoin daily closing prices indexed by date, all calendar days.
    """
    df = yf.download("BTC-USD", start=start, end=end, progress=False, auto_adjust=False)
    close = df["Close"].squeeze().dropna()
    close.name = "Bitcoin_Close"
    return close


def get_prices(
    start: str = START_DATE,
    end: str = END_DATE,
    cache_dir: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Return (gold, bitcoin) daily closing price Series.

    If cache_dir is provided and cached CSVs exist, loads from disk to avoid repeated
    downloads. Otherwise downloads fresh data and optionally saves to disk.

    Parameters
    ----------
    start : str
        Start date.
    end : str
        End date (exclusive).
    cache_dir : str or None
        Directory for caching CSV files. None disables caching.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (gold_prices, bitcoin_prices)
    """
    if cache_dir is not None:
        gold_path = os.path.join(cache_dir, "gold.csv")
        btc_path = os.path.join(cache_dir, "bitcoin.csv")
        if os.path.exists(gold_path) and os.path.exists(btc_path):
            gold = pd.read_csv(gold_path, index_col=0, parse_dates=True).squeeze()
            btc = pd.read_csv(btc_path, index_col=0, parse_dates=True).squeeze()
            gold.name = "Gold_Close"
            btc.name = "Bitcoin_Close"
            return gold, btc

    gold = download_gold(start, end)
    btc = download_bitcoin(start, end)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        gold.to_csv(os.path.join(cache_dir, "gold.csv"))
        btc.to_csv(os.path.join(cache_dir, "bitcoin.csv"))

    return gold, btc


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__))
    gold, btc = get_prices(cache_dir=DATA_DIR)
    print(f"Gold: {len(gold)} records  [{gold.index[0].date()} – {gold.index[-1].date()}]")
    print(f"Bitcoin: {len(btc)} records  [{btc.index[0].date()} – {btc.index[-1].date()}]")
    print(gold.tail(3))
    print(btc.tail(3))
