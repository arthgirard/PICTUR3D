# PICTUR3D
Third version (first public version) of a personal algorithmic SOLANA trading research project with a web interface supported by Flask.

## Getting started
1. Install Python 3.10
2. Install the **TA‑Lib** C library for your platform:
   - **macOS** (Homebrew)
     ```bash
     brew install ta-lib
     ```
   - **Linux** – compile from source
     ```bash
     wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
     tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
     ./configure --prefix=/usr && make
     sudo make install
     ```
   - **Windows** – download and run the installer from [ta-lib.org](https://ta-lib.org/install/)
3. After the library is installed, execute `pip install -r requirements.txt`
4. Run the code in the root folder using `python app.py`
5. Access your interface at `127.0.0.1:5000`

## Features

### Agent
This is a DQN algorithm which is able to save its training weights, scaler and performance. It uses the FinBERT model for sentiment analysis based on news pulled from the GDELT project API and is currently only debugging-ready for backtesting. The paper trading and live trading functions need more work, but it is planned to use the Alpaca Trading API for paper trading and the Kraken API for live trading.

### Web interface
The web interface is pretty minimalistic, presenting live logs and four graphs to measure equity, training loss and bot performance, but also to track where the agent has placed orders over the SOL price chart. The dates of the simulation can be modified along with the number of simulations in a row. It is optional, but the code can also save its results to a folder automatically created in the root of the project.

## Troubleshooting

### Rate limited error
If the Yahoo Finance rate limitation error is encountered, it can be easily fixed by upgrading the module using `pip install yfinance --upgrade` 
