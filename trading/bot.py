# bot.py

import time
import json
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Any, Optional, Dict, List

from trading.data_handler import DataHandler
from trading.sentiment import GDELTNewsClient, SentimentAnalyzer
from trading.agent import TradingAgent
from trading.api_clients import PaperTradingClient, KrakenClient
from config import INITIAL_BALANCE, STOP_LOSS_PCT, TAKE_PROFIT_PCT, DEFAULT_START_DATE, DEFAULT_END_DATE

class TradingBot:
    def __init__(self, mode: str = "backtest", device: str = "cpu", stop_loss_pct: float = STOP_LOSS_PCT, take_profit_pct: float = TAKE_PROFIT_PCT,
                 start_date: str = DEFAULT_START_DATE, end_date: Optional[str] = DEFAULT_END_DATE) -> None:
        self.mode = mode
        self.device = device
        self.data_handler = DataHandler(start_date=start_date, end_date=end_date)
        self.news_client = GDELTNewsClient()
        self.sentiment_analyzer = SentimentAnalyzer(device=self.device)
        self.feature_cols = ['SMA_20', 'RSI', 'MACD', 'Signal', 'Middle_Band', 
                             'Upper_Band', 'Lower_Band', 'Volatility', 'Volume']
        self.input_dim = len(self.feature_cols) + 1  # extra sentiment score
        self.action_space: Dict[int, str] = {0: "hold", 1: "buy_25", 2: "buy_50", 3: "sell_25", 4: "sell_50"}
        self.agent = TradingAgent(input_dim=self.input_dim, action_dim=len(self.action_space), device=self.device)
        self.initial_balance = INITIAL_BALANCE
        self.current_balance = self.initial_balance
        self.current_position = 0.0
        self.trading_fee = 0.001
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.avg_entry_price: Optional[float] = None
        self.sentiment_cache: Dict[str, float] = {}
        if self.mode == "paper":
            self.paper_client = PaperTradingClient(initial_balance=self.initial_balance)
        if self.mode == "live":
            from config import KRAKEN_API_KEY, KRAKEN_API_SECRET
            if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
                raise ValueError("KRAKEN_API_KEY and/or KRAKEN_API_SECRET not found in environment variables.")
            self.kraken_client = KrakenClient(api_key=KRAKEN_API_KEY, api_secret=KRAKEN_API_SECRET)

    def _get_features(self, row: pd.Series) -> np.ndarray:
        tech_features = row[self.feature_cols].values.astype(np.float32)
        date_val = pd.to_datetime(row['Date'])
        if isinstance(date_val, pd.Series):
            date_val = date_val.iloc[0]
        date_str = date_val.strftime("%Y-%m-%d")
        if date_str in self.sentiment_cache:
            sentiment_score = self.sentiment_cache[date_str]
        else:
            if self.mode in ["backtest", "paper"]:
                headlines = self.news_client.fetch_headlines(query="bitcoin", page_size=5, date=date_val)
            else:
                headlines = self.news_client.fetch_headlines(query="bitcoin", page_size=5)
            sentiment_score = self.sentiment_analyzer.compute_sentiment(headlines)
            self.sentiment_cache[date_str] = sentiment_score
        features = np.concatenate([tech_features, np.array([sentiment_score], dtype=np.float32)])
        return features


    def _update_avg_entry_price(self, trade_price: float, btc_amount: float) -> None:
        if self.avg_entry_price is None or self.current_position == 0:
            self.avg_entry_price = trade_price
        else:
            total_cost = self.avg_entry_price * self.current_position
            new_total_cost = total_cost + (trade_price * btc_amount)
            self.avg_entry_price = new_total_cost / (self.current_position + btc_amount)

    def _check_risk_management(self, current_price: float) -> Optional[str]:
        if self.current_position > 0 and self.avg_entry_price is not None:
            if current_price <= self.avg_entry_price * (1 - self.stop_loss_pct):
                logging.info(f"Stop-loss triggered: current price {current_price:.2f} below {self.avg_entry_price * (1 - self.stop_loss_pct):.2f}")
                return "sell_all"
            if current_price >= self.avg_entry_price * (1 + self.take_profit_pct):
                logging.info(f"Take-profit triggered: current price {current_price:.2f} above {self.avg_entry_price * (1 + self.take_profit_pct):.2f}")
                return "sell_all"
        return None

    def _execute_trade(self, action: Any, price: float, mode_client: str) -> None:
        """
        Execute a trade based on the action. Consolidates trade execution logic for backtest, paper, and live modes.
        """
        price = float(price)
        risk_signal = self._check_risk_management(price)
        trade_action = risk_signal if risk_signal == "sell_all" else self.action_space.get(action, "hold")
        if mode_client == "backtest":
            if trade_action.startswith("buy"):
                trade_percentage = 0.25 if "25" in trade_action else 0.50
                trade_amount_usd = self.current_balance * trade_percentage
                btc_bought = (trade_amount_usd * (1 - self.trading_fee)) / price
                self.current_balance -= trade_amount_usd
                self.current_position += btc_bought
                self._update_avg_entry_price(price, btc_bought)
                logging.info(f"Backtest BUY: Spent {trade_amount_usd:.2f} USD to buy {btc_bought:.6f} BTC at {price:.2f}")
            elif trade_action.startswith("sell"):
                trade_percentage = 1.0 if trade_action == "sell_all" else (0.25 if "25" in trade_action else 0.50)
                if self.current_position > 0:
                    btc_to_sell = self.current_position * trade_percentage
                    proceeds = btc_to_sell * price * (1 - self.trading_fee)
                    self.current_balance += proceeds
                    self.current_position -= btc_to_sell
                    if trade_percentage == 1.0:
                        self.avg_entry_price = None
                    logging.info(f"Backtest SELL: Sold {btc_to_sell:.6f} BTC for {proceeds:.2f} USD at {price:.2f}")
        elif mode_client == "paper":
            self.paper_client.place_order(trade_action.split('_')[0], 0.25 if "25" in trade_action else (0.50 if "50" in trade_action else 1.0), price)
        elif mode_client == "live":
            pair = "XBTUSD"
            if trade_action.startswith("buy"):
                trade_percentage = 0.25 if "25" in trade_action else 0.50
                trade_amount_usd = self.current_balance * trade_percentage
                btc_volume = (trade_amount_usd * (1 - self.trading_fee)) / price
                volume = round(btc_volume, 6)
            elif trade_action.startswith("sell"):
                trade_percentage = 1.0 if trade_action == "sell_all" else (0.25 if "25" in trade_action else 0.50)
                btc_volume = self.current_position * trade_percentage
                volume = round(btc_volume, 6)
            else:
                volume = 0
            if volume > 0:
                result = self.kraken_client.place_order(pair=pair, ordertype="market", type_side=trade_action.split('_')[0], volume=volume)
                if result:
                    logging.info(f"Live trading order executed: {result}")

    def run_backtest(self, web_mode: bool = False, stop_event: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        data = self.data_handler.download_data()
        asset_values, dates = [], []
        losses = []
        btc_prices = []
        trade_dates, trade_prices, trade_signals = [], [], []
        self.current_balance = self.initial_balance
        self.current_position = 0.0
        self.avg_entry_price = None
    
        for idx, row in data.iterrows():
            if stop_event is not None and stop_event.is_set():
                logging.info(f"Stop event detected. Exiting simulation loop at iteration {idx}.")
                break
    
            features = self._get_features(row)
            action_idx = self.agent.select_action(features)
            price = float(row['Close']) if not hasattr(row['Close'], 'iloc') else float(row['Close'].iloc[0])
            forced_signal = self._check_risk_management(price)
            action_used = forced_signal if forced_signal == "sell_all" else self.action_space.get(action_idx, "hold")
            if action_used != "hold":
                date_val = pd.to_datetime(row['Date'])
                trade_dates.append(date_val.strftime("%Y-%m-%d"))
                trade_prices.append(price)
                trade_signals.append(action_used)
            self._execute_trade(action_idx if forced_signal is None else forced_signal, price, mode_client="backtest")
            total_asset = self.current_balance + self.current_position * price
            dates.append(pd.to_datetime(row['Date']).strftime("%Y-%m-%d"))
            asset_values.append(total_asset)
            btc_prices.append(price)
            reward = (asset_values[-1] - asset_values[-2]) / asset_values[-2] if idx > 0 else 0
            if forced_signal is None and self.action_space.get(action_idx, "hold") != "hold":
                reward -= 0.001
            self.agent.replay_buffer.add(features, action_idx if forced_signal is None else 0, reward, features, False)
            loss = self.agent.train(batch_size=64)
            if loss is not None:
                losses.append(loss)
            logging.info(f"Date: {dates[-1]}, Action: {action_used}, Total Asset: {total_asset:.2f}")
    
        final_asset = asset_values[-1] if asset_values else self.initial_balance
        net_profit = final_asset - self.initial_balance
        percentage_return = (net_profit / self.initial_balance) * 100
        peak = -float('inf')
        max_drawdown = 0
        for value in asset_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        num_trades = len(trade_dates)
        performance_metrics = {
            "dates": dates,
            "asset_values": asset_values,
            "btc_prices": btc_prices,
            "losses": losses,
            "trade_dates": trade_dates,
            "trade_prices": trade_prices,
            "trade_signals": trade_signals,
            "final_balance": self.current_balance,
            "final_position": self.current_position,
            "net_profit": net_profit,
            "percentage_return": percentage_return,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "finished": True
        }
        try:
            with open("agent_performance.json", "w") as f:
                json.dump(performance_metrics, f)
            logging.info("Performance metrics saved.")
        except Exception as e:
            logging.error(f"Error saving performance metrics: {e}")
    
        if web_mode:
            return performance_metrics
        else:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            axs[0].plot(dates, asset_values, label="Equity")
            axs[0].set_ylabel("Total Asset (USD)")
            axs[0].set_title("Equity Curve")
            axs[0].legend()
            axs[1].plot(dates, btc_prices, label="BTC Price", color="blue")
            for i, d in enumerate(trade_dates):
                if trade_signals[i].startswith("buy"):
                    axs[1].plot(d, trade_prices[i], marker="^", markersize=8, color="green", label="Buy" if i == 0 else "")
                elif trade_signals[i].startswith("sell"):
                    axs[1].plot(d, trade_prices[i], marker="v", markersize=8, color="red", label="Sell" if i == 0 else "")
            axs[1].set_xlabel("Date")
            axs[1].set_ylabel("BTC Price (USD)")
            axs[1].set_title("BTC Price with Trade Signals")
            axs[1].legend()
            plt.tight_layout()
            plt.show()
            self.save_state()

    def run_paper_trading(self) -> None:
        logging.info("Starting paper trading simulation...")
        data = self.data_handler.download_data()
        self.paper_client.balance = self.initial_balance
        self.paper_client.position = 0.0
        self.avg_entry_price = None
        for idx, row in data.iterrows():
            features = self._get_features(row)
            action_idx = self.agent.select_action(features)
            price = float(row['Close']) if not hasattr(row['Close'], 'iloc') else float(row['Close'].iloc[0])
            forced_signal = self._check_risk_management(price)
            action_used = forced_signal if forced_signal == "sell_all" else self.action_space.get(action_idx, "hold")
            self._execute_trade(action_idx if forced_signal is None else forced_signal, price, mode_client="paper")
            self.agent.replay_buffer.add(features, action_idx if forced_signal is None else 0, 0, features, False)
            loss = self.agent.train(batch_size=64)
            if loss:
                logging.info(f"Iteration {idx}, Loss: {loss:.6f}")
            time.sleep(0.1)
        self.save_state()

    def run_live_trading(self) -> None:
        logging.info("Starting live trading mode...")
        data = self.data_handler.download_data()
        self.avg_entry_price = None
        for idx, row in data.iterrows():
            features = self._get_features(row)
            action_idx = self.agent.select_action(features)
            price = float(row['Close']) if not hasattr(row['Close'], 'iloc') else float(row['Close'].iloc[0])
            forced_signal = self._check_risk_management(price)
            action_used = forced_signal if forced_signal == "sell_all" else self.action_space.get(action_idx, "hold")
            self._execute_trade(action_idx if forced_signal is None else forced_signal, price, mode_client="live")
            self.agent.replay_buffer.add(features, action_idx if forced_signal is None else 0, 0, features, False)
            loss = self.agent.train(batch_size=64)
            if loss:
                logging.info(f"Iteration {idx}, Loss: {loss:.6f}")
            time.sleep(1)
        self.save_state()

    def save_state(self) -> None:
        self.agent.save()

    def load_state(self) -> None:
        self.agent.load()
