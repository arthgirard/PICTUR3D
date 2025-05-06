import time
import json
import logging
import math
import numpy as np
import pandas as pd
import datetime
import pickle, os
from typing import Any, Optional, Dict

from trading.data_handler import DataHandler
from trading.sentiment import GDELTNewsClient, SentimentAnalyzer
from trading.agent import TradingAgent
from config import (
    INITIAL_BALANCE,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
)

class TradingBot:
    @staticmethod
    def _to_scalar(x):
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        if isinstance(x, (np.generic, np.ndarray)):
            return float(np.asarray(x).item())
        return float(x)
    
    def __init__(
        self,
        mode: str = "backtest",
        device: str = "cpu",
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
    ) -> None:
        # sentiment cache on disk
        self.sentiment_cache_file = "sentiment_cache.json"
        self.sentiment_cache: Dict[str, float] = {}              # keep type hints

        if os.path.exists(self.sentiment_cache_file):
            try:
                with open(self.sentiment_cache_file, "r") as f:
                    self.sentiment_cache = json.load(f)
                logging.info("Loaded %d cached sentiment scores.",
                             len(self.sentiment_cache))
            except Exception as e:
                logging.warning("Could not load sentiment cache: %s", e)
        self.mode = mode
        self.device = device
        self.scaler = None
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            logging.info("Feature scaler loaded – live data will be normalised.")

        # Data + news
        self.data_handler = DataHandler(start_date=start_date, end_date=end_date)
        self.news_client = GDELTNewsClient()
        self.sentiment_analyzer = SentimentAnalyzer(device=self.device)

        # Agent
        self.feature_cols = [
            "SMA_20", "RSI", "MACD", "Signal",
            "Middle_Band", "Upper_Band", "Lower_Band",
            "Volatility", "Volume", "ATR",
        ]
        self.input_dim = len(self.feature_cols) + 1  # + sentiment score
        self.action_space = {0: "hold", 1: "buy", 2: "sell", 3: "short"}
        self.agent = TradingAgent(
            input_dim=self.input_dim,
            action_dim=len(self.action_space),
            device=self.device,
        )

        # Account state
        self.initial_balance = INITIAL_BALANCE
        self.current_balance = self.initial_balance
        self.current_position = 0.0
        self.trading_fee = 0.001
        self.avg_entry_price: Optional[float] = None

        # Execution clients
        if self.mode == "paper":
            from trading.api_clients import AlpacaPaperClient
            self.paper_client = AlpacaPaperClient(
                api_key=ALPACA_API_KEY,
                api_secret=ALPACA_API_SECRET,
            )
        elif self.mode == "live":
            from trading.api_clients import KrakenClient, KRAKEN_API_KEY, KRAKEN_API_SECRET
            if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
                raise ValueError("Kraken keys are missing.")
            self.kraken_client = KrakenClient(
                api_key=KRAKEN_API_KEY,
                api_secret=KRAKEN_API_SECRET,
            )
            
    # helper: write cache to disk
    def save_sentiment_cache(self) -> None:
        try:
            with open(self.sentiment_cache_file, "w") as f:
                json.dump(self.sentiment_cache, f, indent=2)
            logging.info("Sentiment cache saved (%d dates).",
                         len(self.sentiment_cache))
        except Exception as e:
            logging.error("Error saving sentiment cache: %s", e)

    def _get_features(self, row: pd.Series) -> np.ndarray:
        tech = np.array([self._to_scalar(row[col]) for col in self.feature_cols],
                        dtype=np.float32)
        tech = self._normalize_live(tech)
    
        date_val = pd.to_datetime(row["Date"])
        if isinstance(date_val, pd.Series): 
            date_val = date_val.iloc[0]
        date_str = date_val.strftime("%Y-%m-%d")
        sentiment = self.sentiment_cache.get(date_str)
        if sentiment is None:
            headlines = self.news_client.fetch_headlines("solana", 5,
                                                         pd.to_datetime(date_str))
            sentiment = self.sentiment_analyzer.compute_sentiment(headlines)
            self.sentiment_cache[date_str] = sentiment
        return np.concatenate([tech, [sentiment]], dtype=np.float32)

    # New risk parameter – percentage of the current balance risked per trade
    risk_percentage: float = 0.01  # 1% risk per trade

    def calculate_dynamic_trade_size(self, price: float, atr: float) -> float:
        """
        Compute the dynamic trade size based on the current balance, price, ATR (volatility), and fees.
        The idea is to risk a fixed percentage (risk_percentage) of your balance.
        """
        if isinstance(atr, pd.Series):
            atr = float(atr.iloc[0])
        risk_amount = self.current_balance * self.risk_percentage
        min_stop_loss = 0.005 * price  # minimum stop loss (0.5% of price)
        stop_loss_distance = max(atr, min_stop_loss)
        fee_cost = self.trading_fee * price  # fee cost per unit when buying
        effective_risk_per_unit = stop_loss_distance + fee_cost
        if effective_risk_per_unit == 0:
            return 0.0
        trade_size = risk_amount / effective_risk_per_unit
        return trade_size

    def _check_risk_management(self, current_price: float, atr) -> Optional[str]:
        minimal_position = 1e-6
        if isinstance(atr, pd.Series):
            atr = float(atr.iloc[0])
        # For long positions: trigger sell if the price falls too far below the entry
        if self.current_position > minimal_position and self.avg_entry_price is not None:
            loss_limit = 0.05  # 5% loss threshold
            floor_stop = self.avg_entry_price * (1 - loss_limit)
            if not hasattr(self, 'max_price_since_entry') or self.max_price_since_entry is None:
                self.max_price_since_entry = self.avg_entry_price
            self.max_price_since_entry = max(self.max_price_since_entry, current_price)
            profit_threshold = 0.02  
            if current_price >= self.avg_entry_price * (1 + profit_threshold):
                trailing_factor = 0.5  
                trailing_stop = self.max_price_since_entry - trailing_factor * atr
                min_trailing_stop = self.avg_entry_price * (1 + profit_threshold / 2)
                trailing_stop = max(trailing_stop, min_trailing_stop)
            else:
                trailing_stop = floor_stop
            if current_price <= trailing_stop:
                logging.info(f"(Long risk) Price {current_price:.2f} <= stop {trailing_stop:.2f}")
                return "sell_all"
        # For short positions: trigger cover if the price rises too far above the short entry price
        if self.current_position < -minimal_position and self.avg_entry_price is not None:
            loss_limit = 0.05  # 5% loss threshold for shorts
            ceiling_stop = self.avg_entry_price * (1 + loss_limit)
            if current_price >= ceiling_stop:
                logging.info(f"(Short risk) Price {current_price:.2f} >= ceiling stop {ceiling_stop:.2f}")
                return "cover_all"
        return None
    
    # ---------------------------------------------------------------------
    # apply same µ/σ normalisation on the fly to live features (µ & σ Series)
    # ---------------------------------------------------------------------
    def _normalize_live(self, feat: np.ndarray) -> np.ndarray:
        if not self.scaler:
            return feat.astype(np.float32, copy=True)

        norm = np.empty(len(feat), dtype=np.float32)

        for i, col in enumerate(self.feature_cols):
            v = feat[i]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            if isinstance(v, (np.ndarray, list, tuple)):
                v = np.asarray(v).flatten()[0]
            v = float(v)

            if col in self.scaler:
                stats = self.scaler[col]

                µ = stats["mean"]
                σ = stats["std"]

                if isinstance(µ, pd.Series):
                    µ = µ.iloc[0]
                if isinstance(σ, pd.Series):
                    σ = σ.iloc[0]

                µ = float(µ)
                σ = float(σ) + 1e-8

                norm[i] = (v - µ) / σ
            else:
                norm[i] = v

        return norm

    def _execute_trade(self, action: Any, price: float, atr, mode_client: str, features: np.ndarray) -> str:
        minimal_position = 1e-6
        if isinstance(atr, pd.Series):
            atr = float(atr.iloc[0])
        
        # Check risk management for both long and short positions.
        forced_signal = self._check_risk_management(price, atr)
        if forced_signal is not None:
            trade_action = forced_signal
        else:
            trade_action = self.action_space.get(action, "hold") if not isinstance(action, str) else action
            # If in a long position, enforce profit target to sell partially.
            if self.current_position > minimal_position and self.avg_entry_price is not None:
                profit_target = self.avg_entry_price + 2 * atr
                if price >= profit_target:
                    trade_action = "sell_partial"
        
        # (Optional: if flat, prevent a "sell" action that isn’t meant to be a short.)
        if abs(self.current_position) < minimal_position and trade_action in ["sell", "sell_all", "sell_partial"]:
            logging.info("No long position to sell; holding.")
            trade_action = "hold"
        
        # Execute trade by simulation mode
        if mode_client == "backtest":
            if trade_action == "buy":
                trade_size = self.calculate_dynamic_trade_size(price, atr)
                if trade_size > 0:
                    cost = trade_size * price * (1 + self.trading_fee)
                    if cost > self.current_balance:
                        trade_size = self.current_balance / (price * (1 + self.trading_fee))
                    self.current_balance -= trade_size * price * (1 + self.trading_fee)
                    if self.current_position > minimal_position:
                        total_cost = self.avg_entry_price * self.current_position + price * trade_size
                        self.avg_entry_price = total_cost / (self.current_position + trade_size)
                        self.max_price_since_entry = max(self.max_price_since_entry, price)
                    else:
                        self.avg_entry_price = price
                        self.max_price_since_entry = price
                    self.current_position += trade_size
                    logging.info(f"Backtest BUY: Bought {trade_size:.6f} SOL at {price:.2f} USD")
            elif trade_action in ["sell", "sell_all", "sell_partial"]:
                if self.current_position > minimal_position:
                    trade_size = self.current_position if trade_action == "sell_all" else (self.current_position * 0.5)
                    proceeds = trade_size * price * (1 - self.trading_fee)
                    self.current_balance += proceeds
                    self.current_position -= trade_size
                    logging.info(f"Backtest SELL: Sold {trade_size:.6f} SOL at {price:.2f} USD, proceeds: {proceeds:.2f} USD")
                else:
                    logging.info("No long position to sell.")
            elif trade_action == "short":
                # Open a short position if flat (i.e. no current position)
                if abs(self.current_position) < minimal_position:
                    trade_size = self.calculate_dynamic_trade_size(price, atr)
                    proceeds = trade_size * price * (1 - self.trading_fee)
                    self.current_balance += proceeds
                    self.current_position = -trade_size
                    self.avg_entry_price = price
                    self.max_price_since_entry = price
                    logging.info(f"Backtest SHORT: Shorted {trade_size:.6f} SOL at {price:.2f} USD, proceeds: {proceeds:.2f} USD")
                else:
                    logging.info("Existing position; cannot open a new short trade.")
            elif trade_action == "cover_all":
                # Cover a short position by buying the asset back
                if self.current_position < -minimal_position:
                    trade_size = -self.current_position  # amount needed to cover
                    cost = trade_size * price * (1 + self.trading_fee)
                    if cost > self.current_balance:
                        trade_size = self.current_balance / (price * (1 + self.trading_fee))
                    self.current_balance -= cost
                    self.current_position += trade_size  # This moves the negative position toward zero
                    logging.info(f"Backtest COVER: Covered short of {trade_size:.6f} SOL at {price:.2f} USD, cost: {cost:.2f} USD")
                else:
                    logging.info("No short position to cover.")
            else:
                logging.info("Backtest HOLD: No action taken.")
                
        elif mode_client == "paper":
            cash, pos = self.paper_client.account_state()
        
            if trade_action == "buy":
                qty = self.calculate_dynamic_trade_size(price, atr)
                if qty > 1e-6:
                    self.paper_client.place_order("buy", qty, price)
        
            elif trade_action in ["sell", "sell_all", "sell_partial"]:
                if pos > 1e-6:
                    qty = pos if trade_action == "sell_all" else pos * 0.5
                    self.paper_client.place_order("sell", qty, price)
        
            elif trade_action == "short":
                if abs(pos) < 1e-6:
                    qty = self.calculate_dynamic_trade_size(price, atr)
                    if qty > 1e-6:
                        self.paper_client.open_short(qty, price)
        
            elif trade_action == "cover_all":
                if pos < -1e-6:
                    self.paper_client.close_short(price)
        
            # Sync local copy
            self.current_balance, self.current_position = self.paper_client.account_state()
            if abs(self.current_position) < 1e-6:
                self.avg_entry_price = None
                          
        elif mode_client == "live":
            pair = "SOLUSD"
            if trade_action == "buy":
                trade_size = self.calculate_dynamic_trade_size(price, atr)
                volume = round(trade_size, 6)
                if volume > 0:
                    result = self.kraken_client.place_order(pair=pair, ordertype="market", type_side="buy", volume=volume)
                    if result:
                        logging.info(f"Live BUY order executed: {result}")
            elif trade_action in ["sell", "sell_all", "sell_partial"]:
                if self.current_position > minimal_position:
                    volume = round(self.current_position, 6)
                    result = self.kraken_client.place_order(pair=pair, ordertype="market", type_side="sell", volume=volume)
                    if result:
                        logging.info(f"Live SELL order executed: {result}")
                else:
                    logging.info("Live SELL: No long position to sell.")
            elif trade_action == "short":
                if abs(self.current_position) < minimal_position:
                    trade_size = self.calculate_dynamic_trade_size(price, atr)
                    volume = round(trade_size, 6)
                    # Assuming the live client supports margin short orders via a specific type_side, e.g. "short".
                    result = self.kraken_client.place_order(pair=pair, ordertype="market", type_side="short", volume=volume)
                    if result:
                        logging.info(f"Live SHORT order executed: {result}")
                else:
                    logging.info("Live SHORT: Existing position, cannot open new short.")
            elif trade_action == "cover_all":
                if self.current_position < -minimal_position:
                    volume = round(-self.current_position, 6)
                    # Assuming live API uses a "cover" type to close a short position.
                    result = self.kraken_client.place_order(pair=pair, ordertype="market", type_side="cover", volume=volume)
                    if result:
                        logging.info(f"Live COVER order executed: {result}")
                else:
                    logging.info("Live COVER: No short position to cover.")
        return trade_action
    
    def _build_feature_vector(self, live_row: pd.Series, dt: datetime) -> np.ndarray:
        tech = np.array([self._to_scalar(live_row[col]) for col in self.feature_cols],
                        dtype=np.float32)
        tech = self._normalize_live(tech)
    
        date_key = dt.strftime("%Y-%m-%d")
        if date_key not in self.sentiment_cache:
            headlines = self.news_client.fetch_headlines("solana", 5, dt)
            self.sentiment_cache[date_key] = self.sentiment_analyzer.compute_sentiment(headlines)
        sentiment = self.sentiment_cache[date_key]
        return np.concatenate([tech, [sentiment]], dtype=np.float32)
    
    def run_backtest(self, web_mode: bool = False, stop_event: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        data = self.data_handler.download_data()
        asset_values, dates = [], []
        losses = []
        sol_prices = []  # Price history for SOL
        trade_dates, trade_prices, trade_signals = [], [], []
        self.current_balance = self.initial_balance
        self.current_position = 0.0
        self.avg_entry_price = None
    
        # Get the initial price for the buy-and-hold baseline.
        first_price = float(data.iloc[0]['Close'])
    
        for idx, row in data.iterrows():
            if stop_event is not None and stop_event.is_set():
                logging.info(f"Stop event detected at iteration {idx}.")
                break
    
            atr = row.get("ATR", 0.0)
            features = self._get_features(row)
            if idx < len(data) - 1:
                next_features = self._get_features(data.iloc[idx + 1])
                done = False
            else:
                next_features = features
                done = True
            action_idx = self.agent.select_action(features)
            price = float(row['Close']) if not hasattr(row['Close'], 'iloc') else float(row['Close'].iloc[0])
            forced_signal = self._check_risk_management(price, atr)
    
            final_action = self._execute_trade(
                action_idx if forced_signal is None else forced_signal, 
                price, atr, mode_client="backtest", features=features
            )
    
            if final_action != "hold":
                date_val = pd.to_datetime(row['Date'])
                if isinstance(date_val, pd.Series):
                    date_val = date_val.iloc[0]
                trade_dates.append(date_val.strftime("%Y-%m-%d"))
                trade_prices.append(price)
                trade_signals.append(final_action)
    
            total_asset = self.current_balance + self.current_position * price
            date_val = pd.to_datetime(row['Date'])
            if isinstance(date_val, pd.Series):
                date_val = date_val.iloc[0]
            dates.append(date_val.strftime("%Y-%m-%d"))
            asset_values.append(total_asset)
            sol_prices.append(price)
    
            # Calculate the reward based on log returns.
            if idx > 0:
                reward = math.log(asset_values[-1] / asset_values[-2])
            else:
                reward = 0
    
            # Apply a small penalty for holding to encourage active decision-making.
            if forced_signal is None and self.action_space.get(action_idx, "hold") == "hold":
                reward -= 0.001
    
            # Apply a risk penalty when a position is held (scaled by ATR relative to price).
            if self.current_position > 0:
                risk_penalty = 0.1 * (atr / price)
                reward -= risk_penalty
    
            # Compare against buy-and-hold: compute what a buy-and-hold strategy would yield.
            buy_hold_equity = self.initial_balance * (price / first_price)
            if total_asset > buy_hold_equity:
                # Add a bonus proportional to the logarithmic outperformance.
                bonus = 0.1 * math.log(total_asset / buy_hold_equity)
                reward += bonus
    
            # Removed reward clipping so the learning signal better reflects actual performance.
    
    
            self.agent.replay_buffer.add(
              features,
              action_idx if forced_signal is None else 0,
              reward,
              next_features,
              done,
            )
            loss = self.agent.train(batch_size=64)
            if loss is not None:
                losses.append(loss)
            logging.info(f"Date: {dates[-1]}, Executed Action: {final_action}, Total Asset: {total_asset:.2f}")
    
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
            "sol_prices": sol_prices,
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
            axs[0].plot(dates, asset_values, label="Equity Curve (USD)")
            first_price = sol_prices[0]
            buy_hold_equity_list = [self.initial_balance * (p / first_price) for p in sol_prices]
            axs[0].plot(dates, buy_hold_equity_list, label="Buy & Hold Equity", linestyle="--")
            axs[0].set_ylabel("Total Asset (USD)")
            axs[0].set_title("Equity Curve")
            axs[0].legend()
    
            axs[1].plot(dates, sol_prices, label="SOL Price (USD)", color="blue")
            for i, d in enumerate(trade_dates):
                if trade_signals[i].startswith("buy"):
                    axs[1].plot(d, trade_prices[i], marker="^", markersize=8, color="green", label="Buy" if i == 0 else "")
                elif trade_signals[i].startswith("sell"):
                    axs[1].plot(d, trade_prices[i], marker="v", markersize=8, color="red", label="Sell" if i == 0 else "")
            axs[1].set_xlabel("Date")
            axs[1].set_ylabel("SOL Price (USD)")
            axs[1].set_title("SOL Price with Trade Signals")
            axs[1].legend()
    
            plt.tight_layout()
            plt.show()
            self.save_state()
    
    
    def run_paper_trading_alpaca(self) -> None:
        """Live 1‑min loop with Alpaca paper account."""
        logging.info("Starting Alpaca paper trading (1‑min intervals)…")
        self.paper_client.reset_account(self.initial_balance)
    
        for bar in self.paper_client.stream_bars("SOL/USD"):  # 1‑min bars
            price = bar.close
            atr = self._rolling_atr.update_and_get(bar)        # helper
    
            # Build feature vector
            row = self._realtime_df.update_and_get_row(bar)    # helper
            features = self._build_feature_vector(row, bar.t)
    
            # Pick action
            action_idx = self.agent.select_action(features)
            forced = self._check_risk_management(price, atr)
            final_action = self._execute_trade(
                action_idx if forced is None else forced,
                price, atr, mode_client="paper", features=features,
            )
    
            # Light training
            self.agent.replay_buffer.add(features, action_idx, 0, features, False)
            self.agent.train(batch_size=64)
    
            cash, pos = self.paper_client.account_state()
            logging.info(
                "%s | %s | price=%.2f | cash=%.2f | pos=%.4f",
                bar.t.strftime("%Y-%m-%d %H:%M"),
                final_action.upper(),
                price, cash, pos,
            )
    
    def run_live_trading(self) -> None:
        logging.info("Starting live trading mode...")
        data = self.data_handler.download_data()
        self.avg_entry_price = None
    
        for idx, row in data.iterrows():
            atr = row.get("ATR", 0.0)
            features = self._get_features(row)
            action_idx = self.agent.select_action(features)
            price = float(row['Close']) if not hasattr(row['Close'], 'iloc') else float(row['Close'].iloc[0])
            forced_signal = self._check_risk_management(price, atr)
    
            final_action = self._execute_trade(
                action_idx if forced_signal is None else forced_signal,
                price, atr, mode_client="live", features=features
            )
            
            self.agent.replay_buffer.add(features, action_idx if forced_signal is None else 0, 0, features, False)
            loss = self.agent.train(batch_size=64)
            if loss:
                logging.info(f"Iteration {idx}, Loss: {loss:.6f}")
            logging.info(f"Iteration {idx}, Executed Action: {final_action}, Price: {price:.2f}")
            time.sleep(1)
        self.save_state()
    
    def save_state(self) -> None:
        self.agent.save()
        self.save_sentiment_cache()
    
    def load_state(self) -> None:
        self.agent.load()
