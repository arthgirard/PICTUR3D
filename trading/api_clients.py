# api_clients.py

import time
import logging
import requests
import hashlib
import hmac
import base64
from typing import Tuple, Optional, Dict, Any

class BaseAPIClient:
    """
    Base API client to provide common functionality.
    """
    def _get_nonce(self) -> str:
        """Return a nonce based on current time."""
        return str(int(1000 * time.time()))

    def _sign_request(self, url_path: str, data: str, nonce: str, api_secret: bytes) -> bytes:
        """
        Sign the request using HMAC.
        """
        postdata = (nonce + data).encode()
        message = url_path.encode() + hashlib.sha256(postdata).digest()
        signature = hmac.new(base64.b64decode(api_secret), message, hashlib.sha512)
        return base64.b64encode(signature.digest())


class PaperTradingClient:
    """
    Simulated trading client for paper trading.
    """
    def __init__(self, initial_balance: float = 10000.0) -> None:
        self.balance: float = initial_balance
        self.position: float = 0.0
        self.fee_rate: float = 0.001

    def get_balance(self) -> Tuple[float, float]:
        return self.balance, self.position

    def place_order(self, side: str, percentage: float, price: float) -> None:
        if side == "buy":
            trade_amount_usd = self.balance * percentage
            btc_bought = (trade_amount_usd * (1 - self.fee_rate)) / price
            self.balance -= trade_amount_usd
            self.position += btc_bought
            logging.info(f"Paper Trading BUY: Spent {trade_amount_usd:.2f} USD to buy {btc_bought:.6f} BTC at price {price:.2f}")
        elif side == "sell" and self.position > 0:
            btc_to_sell = self.position * percentage
            proceeds = btc_to_sell * price * (1 - self.fee_rate)
            self.balance += proceeds
            self.position -= btc_to_sell
            logging.info(f"Paper Trading SELL: Sold {btc_to_sell:.6f} BTC for {proceeds:.2f} USD at price {price:.2f}")
        else:
            logging.info("No action taken in paper trading.")


class KrakenClient(BaseAPIClient):
    """
    Kraken API client for live trading.
    """
    def __init__(self, api_key: str, api_secret: str, api_url: str = "https://api.kraken.com") -> None:
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.api_url = api_url

    def _private_request(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        url_path = f"/0/private/{endpoint}"
        url = self.api_url + url_path
        nonce = self._get_nonce()
        data['nonce'] = nonce
        postdata = '&'.join([f"{key}={value}" for key, value in data.items()])
        signature = self._sign_request(url_path, postdata, nonce, self.api_secret).decode()
        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature
        }
        try:
            response = requests.post(url, data=data, headers=headers, timeout=10)
            response.raise_for_status()
            json_resp = response.json()
            if json_resp.get("error"):
                logging.error(f"Kraken API Error ({endpoint}): {json_resp['error']}")
                return None
            return json_resp.get("result", {})
        except Exception as e:
            logging.error(f"Error in Kraken API private request: {e}")
            return None

    def get_account_balance(self) -> Optional[Dict[str, Any]]:
        return self._private_request("Balance", {})

    def place_order(self, pair: str, ordertype: str, type_side: str, volume: float, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        data = {"pair": pair, "ordertype": ordertype, "type": type_side, "volume": volume}
        if price is not None:
            data["price"] = price
        return self._private_request("AddOrder", data)
