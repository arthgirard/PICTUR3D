import os

INITIAL_BALANCE: float = 10000.0
DEFAULT_START_DATE: str = "2020-03-16"  # SOL started on this date
DEFAULT_END_DATE: str = ""

# Kraken API configuration (loaded from environment variables)
KRAKEN_API_KEY: str = os.environ.get("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET: str = os.environ.get("KRAKEN_API_SECRET", "")
