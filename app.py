# app.py

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

app = Flask(__name__)

# Global placeholders for simulation state.
simulation_results = None   # Shared dict for simulation results.
simulation_logs = None      # Shared list for live logs.
stop_event = None           # Shared stop event.

# Use a threading.Lock for shared updates.
from threading import Lock
update_lock = Lock()

# Global simulation process reference.
simulation_process = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def run_backtest_simulation(bot: TradingBot, stop_evt: Any, sim_results: Dict, sim_logs: Any) -> Dict:
    """
    Runs a backtesting simulation using historical data.
    """
    # Clear logs.
    sim_logs[:] = []
    data = bot.data_handler.download_data()
    dates, asset_values, btc_prices = [], [], []
    trade_dates, trade_prices, trade_signals = [], [], []
    losses = []
    
    # Reset account state.
    bot.current_balance = bot.initial_balance
    bot.current_position = 0.0
    bot.avg_entry_price = None

    logging.info("Starting simulation loop over %d rows.", len(data))
    for idx, row in data.iterrows():
        if stop_evt.is_set():
            logging.info("Stop event detected at iteration %d; exiting simulation loop.", idx)
            break

        # Extract and normalize the date value.
        date_val = pd.to_datetime(row['Date'])
        if hasattr(date_val, 'iloc'):  # Fix: if date_val is a Series, take the first element.
            date_val = date_val.iloc[0]
        date_str = date_val.strftime("%Y-%m-%d")
        
        features = bot._get_features(row)
        action_idx = bot.agent.select_action(features)
        price = float(row['Close'].iloc[0]) if isinstance(row['Close'], pd.Series) else float(row['Close'])
        forced_signal = bot._check_risk_management(price)
        action_used = forced_signal if forced_signal == "sell_all" else bot.action_space.get(action_idx, "hold")
        
        if action_used != "hold":
            trade_dates.append(date_str)
            trade_prices.append(price)
            trade_signals.append(action_used)
        
        bot._execute_trade(action_idx if forced_signal is None else forced_signal, price, mode_client="backtest")
        total_asset = bot.current_balance + bot.current_position * price
        asset_values.append(total_asset)
        dates.append(date_str)
        btc_prices.append(price)
        
        log_message = f"{date_str} | Action: {action_used} | Price: {price:.2f} | Total Asset: ${total_asset:.2f}"
        sim_logs.append(log_message)
        
        reward = (asset_values[-1] - asset_values[-2]) / asset_values[-2] if idx > 0 else 0
        if forced_signal is None and bot.action_space.get(action_idx, "hold") != "hold":
            reward -= 0.001
        bot.agent.replay_buffer.add(features, action_idx if forced_signal is None else 0, reward, features, False)
        loss = bot.agent.train(batch_size=64)
        if loss is not None:
            losses.append(loss)
        
        with update_lock:
            sim_results.update({
                "dates": dates,
                "asset_values": asset_values,
                "btc_prices": btc_prices,
                "trade_dates": trade_dates,
                "trade_prices": trade_prices,
                "trade_signals": trade_signals,
                "losses": losses,
                "final_balance": bot.current_balance,
                "final_position": bot.current_position,
                "num_trades": len(trade_dates),
                "percentage_return": ((bot.current_balance + bot.current_position * price - bot.initial_balance) / bot.initial_balance) * 100,
                "net_profit": (bot.current_balance + bot.current_position * price) - bot.initial_balance,
                "max_drawdown": 0,  # Optionally compute incrementally.
                "finished": False
            })
        if bot.mode != "backtest":
            time.sleep(0.2)
    
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
    num_trades = len(trade_dates)
    trade_history = [{"date": d, "price": p, "signal": s} for d, p, s in zip(trade_dates, trade_prices, trade_signals)]
    
    sim_results.update({
        "net_profit": net_profit,
        "percentage_return": percentage_return,
        "max_drawdown": max_drawdown,
        "num_trades": num_trades,
        "trade_history": trade_history,
        "finished": True
    })
    
    bot.save_state()
    logging.info("Simulation completed.")
    return sim_results


def run_simulation(stop_evt: Any, sim_results: Dict, sim_logs: Any, mode: str, start_date: str, end_date: Optional[str],
                   stop_loss_pct: float, take_profit_pct: float) -> None:
    stop_evt.clear()
    bot = TradingBot(mode=mode, device="cpu", stop_loss_pct=stop_loss_pct,
                     take_profit_pct=take_profit_pct, start_date=start_date, end_date=end_date)
    if os.path.exists("agent.pth"):
        bot.load_state()
    if mode in ["backtest", "paper"]:
        run_backtest_simulation(bot, stop_evt, sim_results, sim_logs)
    elif mode == "live":
        bot.run_live_trading()
    else:
        sim_results.update({"error": "Invalid mode"})


def save_performance_metrics(new_metrics: Dict) -> None:
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
    new_metrics["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append(new_metrics)
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
    req = request.json
    mode = req.get("mode", "backtest")
    start_date = req.get("start_date", "2020-01-01")
    end_date = req.get("end_date", None)
    stop_loss_pct = float(req.get("stop_loss_pct", 5)) / 100.0
    take_profit_pct = float(req.get("take_profit_pct", 10)) / 100.0
    stop_event.clear()
    simulation_process = Process(
        target=run_simulation,
        args=(stop_event, simulation_results, simulation_logs, mode, start_date, end_date, stop_loss_pct, take_profit_pct)
    )
    simulation_process.start()
    logging.info("Simulation process started.")
    return jsonify({"status": "Simulation started", "mode": mode})


@app.route("/stop_simulation", methods=["POST"])
def stop_simulation() -> Any:
    stop_simulation_logic()
    save_performance_metrics(dict(simulation_results))
    return jsonify({"status": "Stop signal sent and performance metrics saved."})


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
    save_performance_metrics(dict(simulation_results))
    
    confirmation = request.json.get("confirmation", False)
    if confirmation:
        files_to_delete = ["agent.pth", "replay_buffer.pkl", "scaler.pkl", "performance_history.json"]
        for file in files_to_delete:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    logging.info("Deleted file: %s", file)
                except Exception as e:
                    logging.error("Error deleting file %s: %s", file, e)
        return jsonify({"status": "Agent, replay buffer, scaler, and performance history deleted successfully."})
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
        max_wait = 10  # seconds
        waited = 0
        interval = 0.5  # seconds
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
    freeze_support()  # For Windows support.
    manager = Manager()
    simulation_results = manager.dict()
    simulation_logs = manager.list()
    stop_event = manager.Event()
    app.run(debug=True)
