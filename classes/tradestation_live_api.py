import requests, time, pandas as pd, threading, os, yaml, urllib.parse, webbrowser
from datetime import datetime, timedelta
from PyQt5 import QtCore

BASE_URI = "https://sim-api.tradestation.com"

class TradeStationLiveAPI(QtCore.QObject):
    bar_updated = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(
                    self, symbol, config_path=None, access_token=None, poll_interval=1, 
                    history_days=1, parent=None
                ):
        super().__init__(parent)
        self.symbol = symbol
        self.poll_interval = poll_interval
        self.config_path = config_path or os.path.join("config", "tradestation_config.yaml")
        self.access_token = access_token
        self.headers = None
        self.df = pd.DataFrame(
                                columns=[
                                            "date", "open", "high", "low", "close", 
                                            "volume", "timestamp"
                                        ]
                              )
        self.running = False
        self.thread = None

        if not self.access_token:
            self.access_token = self._authenticate_and_get_token()
        if not self.access_token:
            raise RuntimeError("Failed to obtain TradeStation access_token.")
        self.headers = {'Authorization': f"Bearer {self.access_token}"}

        print(f"Loading last {history_days} days of minute bars for {self.symbol}...")
        self.fetch_last_n_days(days=history_days)
        print(f"Loaded {len(self.df)} historical minute bars for {self.symbol}")

    def _authenticate_and_get_token(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file {self.config_path} not found.")
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        CLIENT_ID = config["CLIENT_ID"]
        CLIENT_SECRET = config["CLIENT_SECRET"]
        REDIRECT_URI = config.get("REDIRECT_URI", "http://localhost")
        AUTH_URL = "https://signin.tradestation.com/authorize"
        TOKEN_URL = "https://signin.tradestation.com/oauth/token"
        params = {
            'response_type': 'code',
            'client_id': CLIENT_ID,
            'redirect_uri': REDIRECT_URI,
            'audience': 'https://api.tradestation.com',
            'scope': 'openid MarketData',
        }
        auth_request_url = AUTH_URL + '?' + urllib.parse.urlencode(params)
        print("Open the following URL in your browser and login:")
        print(auth_request_url)
        webbrowser.open(auth_request_url)
        auth_code = input("Paste code from URL: ").strip()
        data = {
            'grant_type': 'authorization_code',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'code': auth_code,
            'redirect_uri': REDIRECT_URI,
        }
        response = requests.post(TOKEN_URL, data=data)
        tokens = response.json()
        return tokens.get('access_token')

    def fetch_last_n_days(self, days=1):
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        url = (
            f"{BASE_URI}/v3/marketdata/barcharts/{urllib.parse.quote(self.symbol)}"
            f"?unit=Minute&interval=1&firstdate={start_date.isoformat()}Z&lastdate={end_date.isoformat()}Z"
        )
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        bars = data.get("Bars") or data.get("barChartBars") or []
        df = pd.DataFrame([{
            "date": pd.to_datetime(bar["TimeStamp"]),
            "open": float(bar["Open"]),
            "high": float(bar["High"]),
            "low": float(bar["Low"]),
            "close": float(bar["Close"]),
            "volume": float(bar.get("TotalVolume", 0))
        } for bar in bars])
        if not df.empty:
            df['timestamp'] = df['date'].astype('int64') // 10**9
            self.df = df

    def connect(self):
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def disconnect(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_asset_list(self):
        return [self.symbol]

    def get_latest_data(self, symbol=None):
        return self.df
        
    def get_positions(self):
        return {}

    def get_asset_data(self, symbol=None):
        return self.df

    def _poll_loop(self):
        while self.running:
            quote = self._fetch_quote()
            if quote:
                ts = pd.Timestamp(quote.get("TradeTime") \
                    or datetime.utcnow()).replace(second=0, microsecond=0)
                price = quote.get("Last") or quote.get("Bid") or quote.get("Ask") or quote.get("Open")
                volume = quote.get("LastShares", 0)
                if price is not None:
                    self._add_tick(ts, price, volume)
            time.sleep(self.poll_interval)

    def _fetch_quote(self):
        quotes_url = f"{BASE_URI}/v3/marketdata/quotes/{urllib.parse.quote(self.symbol)}"
        resp = requests.get(quotes_url, headers=self.headers)
        if resp.ok:
            data = resp.json()
            return data["Quotes"][0] if "Quotes" in data and data["Quotes"] else None
        return None

    def _add_tick(self, ts, price, volume):
        minute = ts
        price, volume = float(price), float(volume)
        if self.df.empty or minute > self.df['date'].iloc[-1]:
            new_bar = {'date': minute, 'open': price, 'high': price,
                       'low': price, 'close': price, 'volume': volume,
                       'timestamp': int(minute.timestamp())}
            self.df = pd.concat([self.df, pd.DataFrame([new_bar])], ignore_index=True)
        else:
            idx = self.df.index[-1]
            self.df.at[idx, 'high'] = max(self.df.at[idx, 'high'], price)
            self.df.at[idx, 'low'] = min(self.df.at[idx, 'low'], price)
            self.df.at[idx, 'close'] = price
            self.df.at[idx, 'volume'] += volume
        self.bar_updated.emit(self.df)

    def _roll_minute(self):
        if not self.buffer:
            return
        prices = [x['price'] for x in self.buffer]
        volumes = [x['volume'] for x in self.buffer]
        o, h, l, c = prices[0], max(prices), min(prices), prices[-1]
        v = sum(volumes)
        bar = {
            'date': self.current_minute,
            'open': o, 'high': h, 'low': l, 'close': c, 'volume': v
        }
        self.df = pd.concat([self.df, pd.DataFrame([bar])]).tail(2000).reset_index(drop=True)
        if 'timestamp' not in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['date']).astype('int64') // 10**9

    # ---- Not implemented: order/trade methods ----
    def place_order(self, *args, **kwargs):
        print("place_order() called on TradeStationLiveAPI (not supported in live data mode)")
        return None

    def get_order_status(self, *args, **kwargs):
        return None

    def cancel_order(self, *args, **kwargs):
        return None

