# api_clients.py

import time
import logging
import requests
import hashlib
import hmac
import base64
from typing import Tuple, Optional, Dict, Any
import sys, asyncio
# Windows needs a SelectorEventLoop for aiodns / aiohttp combination
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import aiohttp
from datetime import datetime, timezone
import json
from config import (
    INITIAL_BALANCE,
    ALPACA_PAPER_BASE_URL,
)

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

class AlpacaPaperClient:
    """
    Thin wrapper around Alpaca’s paper‑trading REST & WebSocket.
    Keeps only the calls needed by our bot.
    """
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = ALPACA_PAPER_BASE_URL,
        data_url: str = "https://data.alpaca.markets/v2",
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.data_url = data_url.rstrip("/")

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

        # Local mirror of cash & position
        self.cash: float = INITIAL_BALANCE
        self.position: float = 0.0
        self.fee_rate: float = 0.001  # keep fees consistent with backtest

    # --------------------------------------------------------------
    # Account helpers
    # --------------------------------------------------------------
    def account_state(self) -> Tuple[float, float]:
        """Return (cash, position) – updated locally after each order."""
        return self.cash, self.position

    def reset_account(self, starting_balance: float = INITIAL_BALANCE) -> None:
        self.cash = starting_balance
        self.position = 0.0

    # --------------------------------------------------------------
    # Order endpoints
    # --------------------------------------------------------------
    def place_order(self, side: str, qty: float, price: float) -> None:
        """
        Market order simulator (we don’t need limit orders for the bot).
        """
        if side == "buy":
            cost = qty * price * (1 + self.fee_rate)
            if cost > self.cash:
                qty = self.cash / (price * (1 + self.fee_rate))
                cost = qty * price * (1 + self.fee_rate)
            self.cash -= cost
            self.position += qty

        elif side == "sell":
            qty = min(qty, self.position)
            proceeds = qty * price * (1 - self.fee_rate)
            self.cash += proceeds
            self.position -= qty

    # ---- Shorting helpers -----------------------------------------
    def open_short(self, qty: float, price: float) -> None:
        """
        Increase a short position by ‘borrowing’ and selling qty.
        """
        proceeds = qty * price * (1 - self.fee_rate)
        self.cash += proceeds
        self.position -= qty  # negative position

    def close_short(self, price: float) -> None:
        """
        Buy back the entire short position at market.
        """
        if self.position >= 0:
            return  # nothing to cover
        qty = -self.position
        cost = qty * price * (1 + self.fee_rate)
        self.cash -= cost
        self.position = 0.0

    # --------------------------------------------------------------
    # Live bar streaming (crypto, 1‑min) via WebSocket
    # --------------------------------------------------------------
    async def _socket_generator(self, symbol: str, timeframe: str = "1Min"):
        url = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
        auth_msg = {"action": "auth", "key": self.api_key, "secret": self.api_secret}
        sub_msg = {"action": "subscribe", "bars": [symbol]}
        async with aiohttp.ClientSession() as session, session.ws_connect(url) as ws:
            await ws.send_json(auth_msg)
            await ws.send_json(sub_msg)
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.json(loads=json.loads)
                    for event in data:
                        if event.get("T") == "b":  # bar event
                            yield event

    def stream_bars(self, symbol: str):
        """
        Synchronous wrapper around the async websocket generator –
        yields a namedtuple‑like object with .t, .open, .high, .low, .close, .volume
        """
        import threading, queue, collections

        Bar = collections.namedtuple("Bar", "t open high low close volume")
        q: "queue.Queue[Bar]" = queue.Queue()

        async def _runner():
            async for ev in self._socket_generator(symbol):
                raw_t = ev["t"]
                if isinstance(raw_t, (int, float)):
                    ts = datetime.fromtimestamp(raw_t / 1_000_000, tz=timezone.utc)
                else:
                    ts = datetime.fromisoformat(raw_t.replace("Z", "+00:00"))
                q.put(
                    Bar(
                        t=ts,
                        open=float(ev["o"]),
                        high=float(ev["h"]),
                        low=float(ev["l"]),
                        close=float(ev["c"]),
                        volume=float(ev["v"]),
                    )
                )

        threading.Thread(target=lambda: asyncio.run(_runner()), daemon=True).start()

        while True:
            yield q.get()

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
