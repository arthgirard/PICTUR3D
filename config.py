import os

INITIAL_BALANCE: float = 10000.0
DEFAULT_START_DATE: str = "2020-03-16"  # SOL started on this date
DEFAULT_END_DATE: str = ""

# Kraken API configuration (loaded from environment variables)
KRAKEN_API_KEY: str = os.environ.get("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET: str = os.environ.get("KRAKEN_API_SECRET", "")

# Alpaca API configuration (loaded from environment variables)
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "")
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
