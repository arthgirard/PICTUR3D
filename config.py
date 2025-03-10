import os
import datetime

# Flask / App configuration
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key")

# API keys and external services
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET")

# Data settings
DEFAULT_SYMBOL = "BTC-USD"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = datetime.date.today().strftime("%Y-%m-%d")

# Trading settings
INITIAL_BALANCE = 10000.0
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10

# Model & training settings
RL_LEARNING_RATE = 1e-4
RL_GAMMA = 0.99
RL_TAU = 0.005
REPLAY_BUFFER_CAPACITY = 10000
