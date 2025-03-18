import os
import datetime
import pickle
import logging
import pandas as pd
import talib

class DataHandler:
    def __init__(self, symbol="SOL-USD", start_date="2020-01-01", end_date=None,
                 scaler_path="scaler.pkl", normalize: bool = True):
        """
        Initialize the DataHandler.

        Parameters:
            symbol (str): The symbol for which to download historical data.
                          (Changed from "BTC-USD" to "SOL-USD" for Solana.)
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
        df = self.data.copy()
        close = pd.to_numeric(df["Close"].squeeze(), errors='coerce').values.ravel()

        df["SMA_20"] = talib.SMA(close, timeperiod=20)
        df["RSI"] = talib.RSI(close, timeperiod=14)
        df["EMA_12"] = talib.EMA(close, timeperiod=12)
        df["EMA_26"] = talib.EMA(close, timeperiod=26)
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["Signal"] = talib.EMA(df["MACD"].values, timeperiod=9)
        
        df["Middle_Band"] = df["SMA_20"]
        df["Std"] = pd.to_numeric(df["Close"].squeeze(), errors='coerce').rolling(window=20).std()
        df["Upper_Band"] = df["Middle_Band"] + 2 * df["Std"]
        df["Lower_Band"] = df["Middle_Band"] - 2 * df["Std"]
        
        df["Volatility"] = pd.to_numeric(df["Close"].squeeze(), errors='coerce').pct_change().rolling(window=20).std()
        
        self.data = df

    def _handle_missing_values(self) -> None:
        self.data.ffill(inplace=True)
        self.data.bfill(inplace=True)

    def _normalize_features(self) -> None:
        feature_cols = ['SMA_20', 'RSI', 'MACD', 'Signal', 'Middle_Band',
                        'Upper_Band', 'Lower_Band', 'Volatility', 'Volume']
        df = self.data.copy()
        self.scaler = {}
        for col in feature_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std() + 1e-8
                self.scaler[col] = {'mean': mean, 'std': std}
                df[col] = (df[col] - mean) / std

        try:
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logging.info("Saved scaler parameters to %s", self.scaler_path)
        except Exception as e:
            logging.error("Error saving scaler: %s", e)
        self.data = df
