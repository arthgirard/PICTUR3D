# PICTUR3D
Third version (first public version) of a personal algorithmic BTC trading research project with a web interface supported by Flask.

## Getting started
1. Install Python 3.10
2. Add API keys to a ".env" file in /trading **(DO NOT EVER COMMIT)**  
   2a. KRAKEN_API_KEY  
   2b. KRAKEN_API_SECRET  
3. In a terminal, execute `pip install -r requirements.txt`
4. Run the code in the root folder using `python app.py`
5. Access your interface at `127.0.0.1:5000`

## Features

### Agent
The bot saves its training weights, replay buffer, and performance. It uses the FinBERT model for sentiment analysis and is currently only debugging-ready for backtesting. Paper trading and live trading need more work.

### Web interface
The web interface is pretty minimalistic, presenting live logs and four graphs to measure equity, training loss and bot performance, but also to track where the agent has placed orders on the BTC price.
