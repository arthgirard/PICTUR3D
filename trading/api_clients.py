import os
import time
import logging
import requests
import hashlib
import hmac
import base64

class PaperTradingClient:
    def __init__(self, initial_balance=10000.0):
        self.balance = initial_balance
        self.position = 0.0
        self.fee_rate = 0.001

    def get_balance(self):
        return self.balance, self.position

    def place_order(self, side, percentage, price):
        if side == "buy":
            trade_amount_usd = self.balance * percentage
            btc_bought = (trade_amount_usd * (1 - self.fee_rate)) / price
            self.balance -= trade_amount_usd
            self.position += btc_bought
            logging.info("Paper Trading BUY: Spent %.2f USD to buy %.6f BTC at price %.2f", trade_amount_usd, btc_bought, price)
        elif side == "sell" and self.position > 0:
            btc_to_sell = self.position * percentage
            proceeds = btc_to_sell * price * (1 - self.fee_rate)
            self.balance += proceeds
            self.position -= btc_to_sell
            logging.info("Paper Trading SELL: Sold %.6f BTC for %.2f USD at price %.2f", btc_to_sell, proceeds, price)
        else:
            logging.info("No action taken in paper trading.")

class KrakenClient:
    def __init__(self, api_key, api_secret, api_url="https://api.kraken.com"):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.api_url = api_url

    def _get_nonce(self):
        return str(int(1000 * time.time()))

    def _sign(self, url_path, data, nonce):
        postdata = (nonce + data).encode()
        message = url_path.encode() + hashlib.sha256(postdata).digest()
        signature = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(signature.digest())

    def _private_request(self, endpoint, data):
        url_path = f"/0/private/{endpoint}"
        url = self.api_url + url_path
        nonce = self._get_nonce()
        data['nonce'] = nonce
        postdata = '&'.join([f"{key}={value}" for key, value in data.items()])
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._sign(url_path, postdata, nonce).decode()
        }
        response = requests.post(url, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_account_balance(self):
        try:
            response = self._private_request("Balance", {})
            if response.get("error"):
                logging.error("Kraken API Error (Balance): %s", response["error"])
                return None
            return response.get("result", {})
        except Exception as e:
            logging.error("Error fetching Kraken balance: %s", e)
            return None

    def place_order(self, pair, ordertype, type_side, volume, price=None):
        data = {"pair": pair, "ordertype": ordertype, "type": type_side, "volume": volume}
        if price is not None:
            data["price"] = price
        try:
            response = self._private_request("AddOrder", data)
            if response.get("error"):
                logging.error("Kraken API Error (Order): %s", response["error"])
                return None
            logging.info("Kraken order placed: %s", response.get("result"))
            return response.get("result")
        except Exception as e:
            logging.error("Error placing Kraken order: %s", e)
            return None
