import os
import datetime
import pickle
import logging
import pandas as pd
import talib

class DataHandler:
    def __init__(self, symbol="BTC-USD", start_date="2020-01-01", end_date=None,
                 scaler_path="scaler.pkl", normalize: bool = True):
        """
        Initialize the DataHandler.

        Parameters:
            symbol (str): The symbol for which to download historical data.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format. Defaults to today's date.
            scaler_path (str): File path to store/load normalization parameters.
            normalize (bool): If True, normalize selected features.
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")
        self.scaler_path = scaler_path
        self.normalize = normalize
        self.data = None
        self.scaler = None

    def download_data(self) -> pd.DataFrame:
        """
        Download historical data using yfinance, compute technical indicators,
        handle missing values, and apply normalization if enabled.
        """
        logging.info("Downloading historical data for %s", self.symbol)
        import yfinance as yf
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError(f"No data downloaded for symbol: {self.symbol}")
        self.data = data.reset_index()
        self._compute_indicators()
        self._handle_missing_values()
        if self.normalize:
            self._normalize_features()
        return self.data

    def _compute_indicators(self) -> None:
        """
        Compute technical indicators using TA-Lib.
        Indicators computed:
            - SMA_20: 20-period Simple Moving Average
            - RSI: 14-period Relative Strength Index
            - EMA_12: 12-period Exponential Moving Average
            - EMA_26: 26-period Exponential Moving Average
            - MACD: Difference between EMA_12 and EMA_26
            - Signal: 9-period EMA of MACD
            - Middle_Band: using SMA_20 as the middle band
            - Upper_Band and Lower_Band: Bollinger Bands (Middle ± 2 * standard deviation)
            - Volatility: 20-day rolling standard deviation of percentage changes in Close
        """
        df = self.data.copy()
        # Convert Close to a 1D numeric array.
        close = pd.to_numeric(df["Close"].squeeze(), errors='coerce').values.ravel()

        # Core indicators using TA-Lib.
        df["SMA_20"] = talib.SMA(close, timeperiod=20)
        df["RSI"] = talib.RSI(close, timeperiod=14)
        df["EMA_12"] = talib.EMA(close, timeperiod=12)
        df["EMA_26"] = talib.EMA(close, timeperiod=26)
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["Signal"] = talib.EMA(df["MACD"].values, timeperiod=9)
        
        # Bollinger Bands.
        df["Middle_Band"] = df["SMA_20"]
        # Squeeze Close to ensure it's 1-D.
        df["Std"] = pd.to_numeric(df["Close"].squeeze(), errors='coerce').rolling(window=20).std()
        df["Upper_Band"] = df["Middle_Band"] + 2 * df["Std"]
        df["Lower_Band"] = df["Middle_Band"] - 2 * df["Std"]
        
        # Volatility: 20-day rolling standard deviation of percentage change in Close.
        df["Volatility"] = pd.to_numeric(df["Close"].squeeze(), errors='coerce').pct_change().rolling(window=20).std()
        
        self.data = df

    def _handle_missing_values(self) -> None:
        """
        Fill missing values forward and backward.
        """
        self.data.ffill(inplace=True)
        self.data.bfill(inplace=True)

    def _normalize_features(self) -> None:
        """
        Normalize selected features for machine-learning input.
        Features normalized: 'SMA_20', 'RSI', 'MACD', 'Signal', 'Middle_Band',
                             'Upper_Band', 'Lower_Band', 'Volatility', and 'Volume' if available.
        The normalization parameters are saved to scaler_path.
        """
        feature_cols = ['SMA_20', 'RSI', 'MACD', 'Signal', 'Middle_Band',
                        'Upper_Band', 'Lower_Band', 'Volatility', 'Volume']
        df = self.data.copy()
        self.scaler = {}
        for col in feature_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std() + 1e-8  # avoid division by zero
                self.scaler[col] = {'mean': mean, 'std': std}
                df[col] = (df[col] - mean) / std

        try:
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logging.info("Saved scaler parameters to %s", self.scaler_path)
        except Exception as e:
            logging.error("Error saving scaler: %s", e)
        self.data = df
