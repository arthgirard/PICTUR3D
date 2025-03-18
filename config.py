# config.py

import os

# General configuration
INITIAL_BALANCE: float = 10000.0
STOP_LOSS_PCT: float = 0.05
TAKE_PROFIT_PCT: float = 0.10
DEFAULT_START_DATE: str = "2020-03-20"
DEFAULT_END_DATE: str = ""

# Kraken API configuration (loaded from environment variables)
KRAKEN_API_KEY: str = os.environ.get("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET: str = os.environ.get("KRAKEN_API_SECRET", "")
