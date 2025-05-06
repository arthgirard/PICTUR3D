import os
import json
import time
import logging
import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify
from trading.bot import TradingBot
from multiprocessing import Process
from typing import Any, Dict, Optional
import matplotlib
import numpy as np
matplotlib.use('Agg')

app = Flask(__name__)

simulation_results = None
simulation_logs = None
stop_event = None

from threading import Lock
update_lock = Lock()

simulation_process = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
ACTION_WIDTH = 12

def run_backtest_simulation(bot: TradingBot, stop_evt: Any, sim_results: Dict, sim_logs: Any,
                            save_graphs: bool = True, data: Optional[pd.DataFrame] = None,
                            iteration: int = 1, batch_folder: Optional[str] = None,
                            final_iteration: bool = False) -> Dict:
    if data is None:
        data = bot.data_handler.download_data()

    final_date_val = data.iloc[-1]['Date']
    if isinstance(final_date_val, pd.Series):
        final_date_val = final_date_val.iloc[0]
    final_date = pd.to_datetime(final_date_val)
    final_date_str = final_date.strftime("%Y-%m-%d")


    dates, asset_values, sol_prices = [], [], []
    trade_dates, trade_prices, trade_signals = [], [], []
    losses = []

    bot.current_balance = bot.initial_balance
    bot.current_position = 0.0
    bot.avg_entry_price = None

    logging.info("Starting simulation loop over %d rows.", len(data))

    for idx, row in data.iterrows():
        if stop_evt.is_set():
            logging.info("Stop event detected at row %d; exiting simulation loop.", idx)
            break

        date_val = row['Date']
        if isinstance(date_val, pd.Series):
            date_val = date_val.iloc[0]
        date_val = pd.to_datetime(date_val)
        price = float(row['Close'].iloc[0]) if isinstance(row['Close'], pd.Series) else float(row['Close'])
        atr = row.get("ATR", 0.0)

        # Get current and next state features
        current_features = bot._get_features(row)
        if idx < len(data) - 1:
            next_row = data.iloc[idx + 1]
            next_features = bot._get_features(next_row)
            done = False
        else:
            next_features = current_features
            done = True

        # Decide action
        action_idx = bot.agent.select_action(current_features)
        forced_signal = bot._check_risk_management(price, atr)
        final_action = bot._execute_trade(
            action_idx if forced_signal is None else forced_signal,
            price, atr, mode_client="backtest", features=current_features
        )

        # Log trade
        if final_action != "hold":
            trade_dates.append(date_val.strftime("%Y-%m-%d"))
            trade_prices.append(price)
            trade_signals.append(final_action)

        # Compute total asset value
        total_asset = bot.current_balance + bot.current_position * price
        dates.append(date_val.strftime("%Y-%m-%d"))
        asset_values.append(total_asset)
        sol_prices.append(price)

        # Compute reward: asset delta (optionally include drawdown penalty)
        if len(asset_values) > 1:
            prev_asset = asset_values[-2]
            drawdown_penalty = max(0, (prev_asset - total_asset) / prev_asset)
            reward = (total_asset - prev_asset) / prev_asset - 0.5 * drawdown_penalty
        else:
            reward = 0.0

        # Store experience
        bot.agent.replay_buffer.add(current_features, action_idx, reward, next_features, done)

        # Train agent
        loss = bot.agent.train(batch_size=64)
        if loss is not None:
            losses.append(loss)

        # Buy & Hold tracking
        if sol_prices:
            first_price = sol_prices[0]
            current_buy_hold = [bot.initial_balance * (p / first_price) for p in sol_prices]
        else:
            current_buy_hold = []

        # Logging
        action_fmt = f"{final_action:<{ACTION_WIDTH}}" 
        price_fmt  = f"{price:>8.2f}"                 
        log_message = (
            f"{dates[-1]} | ACTION: {action_fmt} | "
            f"PRICE: {price_fmt} | TOTAL ASSETS: ${total_asset:,.2f}"
        )
        sim_logs.append(log_message)

        with update_lock:
            sim_results.update({
                "dates": dates,
                "asset_values": asset_values,
                "sol_prices": sol_prices,
                "buy_hold_equity": current_buy_hold,
                "trade_dates": trade_dates,
                "trade_prices": trade_prices,
                "trade_signals": trade_signals,
                "losses": losses,
                "final_balance": bot.current_balance,
                "final_position": bot.current_position,
                "num_trades": len(trade_dates),
                "percentage_return": ((total_asset - bot.initial_balance) / bot.initial_balance) * 100,
                "net_profit": total_asset - bot.initial_balance,
                "max_drawdown": 0,
                "finished": False,
                "iteration": iteration
            })

        if dates[-1] == final_date_str:
            logging.info("Final date reached: %s. Finalizing simulation iteration.", final_date_str)
            break

        if bot.mode != "backtest":
            time.sleep(0.2)

    # Post-simulation metrics
    final_asset = asset_values[-1] if asset_values else bot.initial_balance
    net_profit = final_asset - bot.initial_balance
    percentage_return = (net_profit / bot.initial_balance) * 100
    peak = -float('inf')
    max_drawdown = 0
    for value in asset_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak else 0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    trade_history = [{"date": d, "price": p, "signal": s} for d, p, s in zip(trade_dates, trade_prices, trade_signals)]

    sim_results.update({
        "net_profit": net_profit,
        "percentage_return": percentage_return,
        "max_drawdown": max_drawdown,
        "num_trades": len(trade_dates),
        "trade_history": trade_history,
        "buy_hold_equity": current_buy_hold,
        "finished": final_iteration,
        "iteration": iteration
    })

    bot.save_state()
    save_performance_metrics(dict(sim_results))

    if save_graphs:
        try:
            import matplotlib.pyplot as plt
            from datetime import datetime
            iteration_folder = os.path.join(batch_folder, f"iteration_{iteration}")
            os.mkdir(iteration_folder)

            dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

            # Equity chart with simulation and Buy & Hold curves.
            fig, ax = plt.subplots()
            ax.plot(dates_dt, asset_values, label="Equity Curve (USD)")
            ax.plot(dates_dt, current_buy_hold, label="Buy & Hold Equity", linestyle="--")
            ax.set_xlabel("Date")
            ax.set_ylabel("Total Asset (USD)")
            ax.set_title("Equity Curve")
            ax.legend()
            fig.autofmt_xdate()
            fig.savefig(os.path.join(iteration_folder, "equity_chart.png"))
            plt.close(fig)

            trade_dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in trade_dates]
            fig, ax = plt.subplots()
            ax.plot(dates_dt, sol_prices, label="SOL Price (USD)", color="blue")
            for i, d in enumerate(trade_dates_dt):
                if trade_signals[i].lower().startswith("buy"):
                    ax.plot([d], [trade_prices[i]], marker="^", markersize=8, color="green", label="Buy" if i == 0 else "")
                elif trade_signals[i].lower().startswith("sell"):
                    ax.plot([d], [trade_prices[i]], marker="v", markersize=8, color="red", label="Sell" if i == 0 else "")
            ax.set_xlabel("Date")
            ax.set_ylabel("SOL Price (USD)")
            ax.set_title("SOL Price with Trades")
            ax.legend()
            fig.autofmt_xdate()
            fig.savefig(os.path.join(iteration_folder, "sol_price_chart.png"))
            plt.close(fig)

            if losses:
                fig, ax = plt.subplots()
                ax.plot(range(1, len(losses)+1), losses, label="Training Loss")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss")
                ax.legend()
                fig.savefig(os.path.join(iteration_folder, "loss_chart.png"))
                plt.close(fig)

            stats = {
                "Final Balance": bot.current_balance,
                "Final Position": bot.current_position,
                "Net Profit": net_profit,
                "Percentage Return": percentage_return,
                "Number of Trades": num_trades,
                "Max Drawdown": max_drawdown
            }
            df_stats = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
            df_stats.to_excel(os.path.join(iteration_folder, "simulation_stats.xlsx"), index=False)

            logging.info("Simulation iteration %d results saved to folder.", iteration)
        except Exception as e:
            logging.error("Error while saving simulation outputs: %s", e)

    return sim_results

# --------------------------------------------------------------------------- #
# Real‑time Alpaca paper‑trading loop (1‑min crypto bars)                    #
# --------------------------------------------------------------------------- #
def run_paper_trading(
        bot: TradingBot,
        stop_evt,
        paper_results,
        paper_logs,
        symbol: str = "SOL/USD",
        duration_minutes: Optional[int] = None
) -> None:
    """
    Mirrors the back‑test logic but drives on live streaming bars pulled from
    Alpaca’s crypto WebSocket.  Results & logs are pushed into the shared
    Manager dict/list so the Flask UI can poll them exactly as for back‑tests.
    """
    import pandas as pd, talib
    from datetime import datetime, timezone, timedelta
    from collections import deque

    start_ts = datetime.now(tz=timezone.utc)
    df_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    history = pd.DataFrame(columns=df_cols)          # rolling history window

    # Series passed to the UI
    dates, asset_vals, sol_prices = [], [], []
    trade_dates, trade_prices, trade_sigs = [], [], []
    first_price = None

    bot.paper_client.reset_account(bot.initial_balance)

    log = logging.getLogger("paper_loop")
    log.info("🚀  Paper‑trading loop started for %s", symbol)

    # --------------------------------------------------------------------- #
    # Stream 1‑min bars forever (or until stop_evt / duration reached)     #
    # --------------------------------------------------------------------- #
    for bar in bot.paper_client.stream_bars(symbol):
        if stop_evt.is_set():
            log.info("Stop event received – shutting paper loop down.")
            break
        if duration_minutes and \
           datetime.now(tz=timezone.utc) - start_ts >= timedelta(minutes=duration_minutes):
            log.info("Duration limit reached – shutting paper loop down.")
            break

        # ─── store current bar ─────────────────────────────────────────────
        row_dict = {
            "Date":   bar.t,
            "Open":   float(bar.open),
            "High":   float(bar.high),
            "Low":    float(bar.low),
            "Close":  float(bar.close),
            "Volume": float(bar.volume),
        }
        history.loc[len(history)] = row_dict    

        # We always try to compute indicators, even if there are only a few rows.
        # TA‑Lib will return NaN until it has enough data – we simply fill those with 0.
        close = history["Close"].astype(float).to_numpy()
        high  = history["High"].astype(float).to_numpy()
        low   = history["Low"].astype(float).to_numpy()

        # Safe helpers ------------------------------------------------------
        def last_or_zero(obj):
            """
            Return the last finite value of a 1‑D NumPy array *or* pandas Series,
            falling back to 0.0 for empty / all‑nan input.
            """
            if obj is None or len(obj) == 0:
                return 0.0
            if isinstance(obj, pd.Series):
                val = obj.iloc[-1]
            else:                          # NumPy array / list
                val = obj[-1]
            return 0.0 if (pd.isna(val) or np.isnan(val)) else float(val)


        sma20 = last_or_zero(talib.SMA(close, 20))
        rsi14 = last_or_zero(talib.RSI(close, 14))
        macd, macd_sig, _ = talib.MACD(close, 12, 26, 9)
        macd_v   = last_or_zero(macd)
        macd_sig = last_or_zero(macd_sig)
        std20 = last_or_zero(pd.Series(close).rolling(20).std())
        mid_band   = sma20
        upper_band = mid_band + 2 * std20
        lower_band = mid_band - 2 * std20
        volat   = last_or_zero(pd.Series(close).pct_change().rolling(20).std())
        atr14   = last_or_zero(talib.ATR(high, low, close, 14))

        live_row = pd.Series({
            "SMA_20": sma20,
            "RSI": rsi14,
            "MACD": macd_v,
            "Signal": macd_sig,
            "Middle_Band": mid_band,
            "Upper_Band": upper_band,
            "Lower_Band": lower_band,
            "Volatility": volat,
            "Volume": history["Volume"].iloc[-1],
            "ATR": atr14,
        })


        # ----- decision, execution, learning ----------------------------
        features     = bot._build_feature_vector(live_row, bar.t)
        action_idx   = bot.agent.select_action(features)
        forced_sig   = bot._check_risk_management(bar.close, atr14)
        final_action = bot._execute_trade(
            action_idx if forced_sig is None else forced_sig,
            bar.close,
            atr14,
            mode_client="paper",
            features=features,
        )

        # ----- bookkeeping ---------------------------------------------
        if final_action != "hold":
            trade_dates.append(bar.t.strftime("%Y-%m-%d %H:%M"))
            trade_prices.append(bar.close)
            trade_sigs.append(final_action)

        cash, pos = bot.paper_client.account_state()
        tot_asset = cash + pos * bar.close
        label     = bar.t.strftime("%Y-%m-%d %H:%M")
        dates.append(label)
        asset_vals.append(tot_asset)
        sol_prices.append(bar.close)

        if first_price is None:
            first_price = bar.close
        buy_hold_equity = [bot.initial_balance * (p / first_price) for p in sol_prices]

        log_line = f"{label} | ACTION: {final_action.upper()} | price={bar.close:.2f} | equity=${tot_asset:.2f}"
        paper_logs.append(log_line)

        with update_lock:            # push snapshot to UI every bar
            paper_results.update({
                "dates": dates,
                "asset_values": asset_vals,
                "sol_prices": sol_prices,
                "buy_hold_equity": buy_hold_equity,
                "trade_dates": trade_dates,
                "trade_prices": trade_prices,
                "trade_signals": trade_sigs,
                "losses": [],
                "final_balance": cash,
                "final_position": pos,
                "num_trades": len(trade_dates),
                "percentage_return": ((tot_asset - bot.initial_balance) / bot.initial_balance) * 100,
                "net_profit": tot_asset - bot.initial_balance,
                "finished": False,
            })

    # ---------- exit -------------------------------------------
    with update_lock:
        paper_results["finished"] = True
    log.info("Paper‑trading loop finished")

def _paper_worker(stop_evt, sim_results, sim_logs, symbol, duration):
    """
    Spawned in a separate **process** by /start_paper.
    It now loads the pre-trained agent weights so paper trading
    uses exactly the same policy as backtesting.
    """
    bot = TradingBot(mode="paper", device="cpu")

    # load agent.pth if exist
    try:
        bot.load_state()
        # zero exploration
        bot.agent.epsilon = bot.agent.epsilon_min
    except FileNotFoundError:
        import logging
        logging.warning("agent.pth not found, starting fresh.")

    run_paper_trading(
        bot,
        stop_evt,
        sim_results,
        sim_logs,
        symbol=symbol,
        duration_minutes=duration,
    )

def run_simulation(stop_evt: Any, sim_results: Dict, sim_logs: Any, mode: str, start_date: str, end_date: Optional[str],
                   number_of_simulations: int, save_graphs: bool) -> None:
    stop_evt.clear()

    # Create a persistent TradingBot instance to retain training across iterations.
    bot = TradingBot(mode=mode, device="cpu", start_date=start_date, end_date=end_date)
    if os.path.exists("agent.pth"):
        bot.load_state()
    data = bot.data_handler.download_data()

    batch_folder = None
    if save_graphs:
        root_folder = os.path.join(os.getcwd(), "backtest_results")
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
        batch_folder = os.path.join(root_folder, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.mkdir(batch_folder)

    sim_results["finished"] = False
    sim_results["total_simulations"] = number_of_simulations

    for i in range(number_of_simulations):
        if stop_evt.is_set():
            logging.info("Stop event detected before iteration %d; exiting.", i+1)
            break

        sim_results["iteration"] = i + 1
        sim_logs[:] = []

        logging.info("Starting simulation iteration %d of %d", i+1, number_of_simulations)
        final_iteration = (i == number_of_simulations - 1)
        run_backtest_simulation(bot, stop_evt, sim_results, sim_logs, save_graphs, data=data,
                                iteration=i+1, batch_folder=batch_folder, final_iteration=final_iteration)

        if stop_evt.is_set():
            logging.info("Stop event detected after iteration %d; exiting.", i+1)
            break
        else:
            stop_evt.clear()

    if not stop_evt.is_set():
        sim_results["finished"] = True
    time.sleep(1)

def save_performance_metrics(new_metrics: Dict) -> None:
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "net_profit": new_metrics.get("net_profit", 0)
    }
    history_file = "performance_history.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception as e:
            logging.error("Error reading performance history: %s", e)
            history = []
    else:
        history = []
    history.append(record)
    try:
        with open(history_file, "w") as f:
            json.dump(history, f, indent=4)
        logging.info("Performance history updated.")
    except Exception as e:
        logging.error("Error saving performance history: %s", e)

@app.route("/")
def index() -> Any:
    return render_template("index.html")

@app.route("/start_simulation", methods=["POST"])
def start_simulation() -> Any:
    global simulation_process
    simulation_results.clear()
    simulation_logs[:] = []
    stop_event.clear()

    req = request.json
    mode = req.get("mode", "backtest")
    start_date = req.get("start_date", "2020-03-16")
    end_date = req.get("end_date", None)
    number_of_simulations = int(req.get("number_of_simulations", 1))
    save_graphs = bool(req.get("save_graphs", False))

    simulation_process = Process(
        target=run_simulation,
        args=(stop_event, simulation_results, simulation_logs, mode, start_date, end_date,
              number_of_simulations, save_graphs)
    )
    simulation_process.start()
    logging.info("Simulation process started.")
    return jsonify({"status": "Simulation started", "mode": mode})

@app.route("/stop_simulation", methods=["POST"])
def stop_simulation() -> Any:
    stop_simulation_logic()
    return jsonify({"status": "Stop signal sent and performance metrics saved."})

@app.route("/start_paper", methods=["POST"])
def start_paper():
    """
    Launch a real‑time Alpaca paper‑trading session in a background process.
    Body JSON accepts:
        - symbol   : crypto symbol, default "SOL/USD"
        - duration : minutes to run (omit or null to run until /stop_simulation)
    """
    global simulation_process
    simulation_results.clear()
    simulation_logs[:] = []
    stop_event.clear()

    req       = request.get_json(force=True, silent=True) or {}
    symbol    = req.get("symbol", "SOL/USD")
    duration  = req.get("duration")          # may be None / null
    duration  = int(duration) if duration else None

    simulation_process = Process(
        target=_paper_worker,
        args=(stop_event, simulation_results, simulation_logs, symbol, duration),
    )
    simulation_process.start()
    logging.info("Started paper‑trading process for %s (duration=%s)", symbol, duration)
    return jsonify({"status": "Paper trading started", "symbol": symbol})


@app.route("/results", methods=["GET"])
def results_route() -> Any:
    return jsonify(dict(simulation_results))

@app.route("/live_logs", methods=["GET"])
def live_logs_route() -> Any:
    return jsonify(list(simulation_logs))

@app.route("/agent_performance", methods=["GET"])
def agent_performance() -> Any:
    history_file = "performance_history.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
            return jsonify(data if isinstance(data, list) else [])
        except Exception as e:
            logging.error("Error reading performance history: %s", e)
            return jsonify({"error": "Error reading performance history"}), 500
    else:
        return jsonify([])

@app.route("/delete_agent", methods=["POST"])
def delete_agent() -> Any:
    global simulation_process
    stop_simulation_logic()

    confirmation = request.json.get("confirmation", False)
    if confirmation:
        files_to_delete = ["agent.pth", "scaler.pkl", "performance_history.json"]
        for file in files_to_delete:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    logging.info("Deleted file: %s", file)
                except Exception as e:
                    logging.error("Error deleting file %s: %s", file, e)
        return jsonify({"status": "Agent, scaler, and performance history deleted successfully."})
    else:
        return jsonify({"status": "Deletion canceled."}), 400

@app.route("/clear_logs", methods=["POST"])
def clear_logs() -> Any:
    simulation_logs[:] = []
    return jsonify({"status": "Logs cleared"})

def stop_simulation_logic() -> None:
    logging.info("Stop simulation logic initiated.")
    stop_event.set()
    global simulation_process
    if simulation_process is not None:
        max_wait = 10
        waited = 0
        interval = 0.5
        try:
            while simulation_process.is_alive() and waited < max_wait:
                logging.info("Waiting for simulation process to stop... (waited %.1f sec)", waited)
                time.sleep(interval)
                waited += interval
        except Exception as e:
            logging.error("Error while waiting for simulation process to stop: %s", e)
        try:
            if simulation_process.is_alive():
                logging.warning("Simulation process did not exit after waiting %s seconds.", max_wait)
            else:
                logging.info("Simulation process stopped after %.1f seconds.", waited)
        except Exception as e:
            logging.error("Error checking if simulation process is alive: %s", e)
        simulation_process = None
    else:
        logging.info("No simulation process to stop.")

if __name__ == '__main__':
    from multiprocessing import freeze_support, Manager
    freeze_support()
    manager = Manager()
    simulation_results = manager.dict()
    simulation_logs = manager.list()
    stop_event = manager.Event()
    app.run(debug=True)
