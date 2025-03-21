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
matplotlib.use('Agg')

app = Flask(__name__)

simulation_results = None
simulation_logs = None
stop_event = None

from threading import Lock
update_lock = Lock()

simulation_process = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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

        atr = row.get("ATR", 0.0)
        features = bot._get_features(row)
        action_idx = bot.agent.select_action(features)
        price = float(row['Close'].iloc[0]) if isinstance(row['Close'], pd.Series) else float(row['Close'])
        forced_signal = bot._check_risk_management(price, atr)
        action_used = forced_signal if forced_signal == "sell_all" else bot.action_space.get(action_idx, "hold")
        
        if action_used != "hold":
            date_val = pd.to_datetime(row['Date'])
            if isinstance(date_val, pd.Series):
                date_val = date_val.iloc[0]
            trade_dates.append(date_val.strftime("%Y-%m-%d"))
            trade_prices.append(price)
            trade_signals.append(action_used)
        
        bot._execute_trade(action_idx if forced_signal is None else forced_signal, price, atr, mode_client="backtest")
        total_asset = bot.current_balance + bot.current_position * price
        date_val = pd.to_datetime(row['Date'])
        if isinstance(date_val, pd.Series):
            date_val = date_val.iloc[0]
        dates.append(date_val.strftime("%Y-%m-%d"))
        asset_values.append(total_asset)
        sol_prices.append(price)
        
        log_message = f"{dates[-1]} | ACTION: {action_used} | PRICE: {price:.2f} | TOTAL ASSET: ${total_asset:.2f}"
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
                "sol_prices": sol_prices,
                "trade_dates": trade_dates,
                "trade_prices": trade_prices,
                "trade_signals": trade_signals,
                "losses": losses,
                "final_balance": bot.current_balance,
                "final_position": bot.current_position,
                "num_trades": len(trade_dates),
                "percentage_return": ((bot.current_balance + bot.current_position * price - bot.initial_balance) / bot.initial_balance) * 100,
                "net_profit": (bot.current_balance + bot.current_position * price) - bot.initial_balance,
                "max_drawdown": 0,
                "finished": False,
                "iteration": iteration
            })
        
        if dates[-1] == final_date_str:
            logging.info("Final date reached: %s. Finalizing simulation iteration.", final_date_str)
            break
        
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
    
    if final_iteration:
        sim_results.update({
            "net_profit": net_profit,
            "percentage_return": percentage_return,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "trade_history": trade_history,
            "finished": True,
            "iteration": iteration
        })
    else:
        sim_results.update({
            "net_profit": net_profit,
            "percentage_return": percentage_return,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "trade_history": trade_history,
            "finished": False,
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
            
            fig, ax = plt.subplots()
            ax.plot(dates_dt, asset_values, label="Equity Curve (USD)")
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
    
    logging.info("Simulation iteration %d completed.", iteration)
    return sim_results

def run_simulation(stop_evt: Any, sim_results: Dict, sim_logs: Any, mode: str, start_date: str, end_date: Optional[str],
                   number_of_simulations: int, save_graphs: bool) -> None:
    stop_evt.clear()
    
    bot_for_data = TradingBot(mode=mode, device="cpu", start_date=start_date, end_date=end_date)
    data = bot_for_data.data_handler.download_data()
    
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
        bot = TradingBot(mode=mode, device="cpu", start_date=start_date, end_date=end_date)
        if os.path.exists("agent.pth"):
            bot.load_state()
        
        final_iteration = (i == number_of_simulations - 1)
        run_backtest_simulation(bot, stop_evt, sim_results, sim_logs, save_graphs, data=data,
                                iteration=i+1, batch_folder=batch_folder, final_iteration=final_iteration)
        
        if stop_evt.is_set():
            logging.info("Stop event detected after iteration %d; exiting.", i+1)
            break
        
        if not stop_evt.is_set():
            stop_evt.clear()
    
    if not stop_evt.is_set():
        sim_results["finished"] = True
    time.sleep(1)

def save_performance_metrics(new_metrics: Dict) -> None:
    # Only record the final net profit and the current timestamp.
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
