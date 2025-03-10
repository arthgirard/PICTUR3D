import os
import datetime
import pickle
import logging
import pandas as pd
import yfinance as yf

class DataHandler:
    def __init__(self, symbol="BTC-USD", start_date="2020-01-01", end_date=None, scaler_path="scaler.pkl"):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.date.today().strftime("%Y-%m-%d")
        self.scaler_path = scaler_path
        self.data = None
        self.scaler = None

    def download_data(self):
        logging.info("Downloading historical data for %s", self.symbol)
        try:
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        except Exception as e:
            logging.error("Error downloading data: %s", e)
            raise

        if data.empty:
            raise ValueError(f"No data downloaded for symbol: {self.symbol}")
        self.data = data.reset_index()
        self._compute_indicators()
        self._handle_missing_values()
        self._normalize_features()
        return self.data

    def _compute_indicators(self):
        df = self.data.copy()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Middle_Band'] = df['Close'].rolling(window=20).mean()
        df['Std'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['Middle_Band'] + (df['Std'] * 2)
        df['Lower_Band'] = df['Middle_Band'] - (df['Std'] * 2)
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        self.data = df

    def _handle_missing_values(self):
        self.data.ffill(inplace=True)
        self.data.bfill(inplace=True)

    def _normalize_features(self):
        feature_cols = ['SMA_20', 'RSI', 'MACD', 'Signal', 'Middle_Band', 
                        'Upper_Band', 'Lower_Band', 'Volatility', 'Volume']
        df = self.data.copy()
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logging.info("Loaded scaler parameters from %s", self.scaler_path)
            except Exception as e:
                logging.error("Error loading scaler: %s", e)
                self.scaler = None
        if self.scaler is None:
            self.scaler = {}
            for col in feature_cols:
                self.scaler[col] = {'mean': df[col].mean(), 'std': df[col].std() + 1e-8}
            try:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logging.info("Saved scaler parameters to %s", self.scaler_path)
            except Exception as e:
                logging.error("Error saving scaler: %s", e)
        for col in feature_cols:
            df[col] = (df[col] - self.scaler[col]['mean']) / self.scaler[col]['std']
        self.data = df
