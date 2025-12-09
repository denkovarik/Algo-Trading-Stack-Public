# classes/tradestation_live_api.py
import requests
import time
import pandas as pd
import threading
import os
import yaml
import urllib.parse
import json
import webbrowser
from datetime import datetime, timedelta, timezone
from PyQt5 import QtCore

# === API Interface & Data Structures ===
from classes.API_Interface import APIInterface, Position, MetaQObjectABC
from classes.API_Interface import round_to_tick

BASE_URI = "https://sim-api.tradestation.com"
CSV_PATH = "TradeStation/data/PL/5min/PL_live.csv"
EQUITY_FILE = "TradeStation/data/equity_5min.csv"


class TradeStationLiveAPI(QtCore.QObject, APIInterface, metaclass=MetaQObjectABC):
    """
    Live connector to TradeStation SIM API.
    Sends entry + StopMarket immediately using ParentOrderID.
    FULLY SYNCHRONIZED with broker position & equity.
    Auto-tightens stop 60s after fill.
    """
    bar_updated = QtCore.pyqtSignal(pd.DataFrame)
    equity_updated = QtCore.pyqtSignal(float, str)

    # --------------------------------------------------------------------- #
    # INITIALIZATION
    # --------------------------------------------------------------------- #
    def __init__(self, symbol="@PL", config_path=None, access_token=None,
                 poll_interval=20, bar_interval=5, parent=None, account_id=None):
        super().__init__(parent)
        self.symbol = symbol
        self.account_id = account_id
        self.poll_interval = poll_interval
        self.bar_interval = bar_interval
        self.config_path = config_path or "TradeStation/config/tradestation_config.yaml"
        self.access_token = access_token
        self.headers = None
        self.df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "timestamp"])
        self.running = False
        self.thread = None
        self.last_minute_written = None

        os.makedirs("TradeStation/data", exist_ok=True)
        self.equity_file = EQUITY_FILE

        print(f"[INIT] Using CSV: {CSV_PATH}")
        print(f"[INIT] Symbol: {symbol} | Bar: {bar_interval}min | Poll: {poll_interval}s")

        # ----------------------------------------------------------------- #
        # Symbol configuration
        # ----------------------------------------------------------------- #
        self.data_symbol = self.symbol
        self.trading_symbol = "PLX5"
        self.order_timeout = 30
        self.tick_size = 0.1
        self.tick_value = 5.0  # $5 per 0.1 move

        # === NEW: For BaseStrategyBot ===
        self.config = {}
        self.assets = []

        # === LOAD CONFIG (one line) ===
        config_path = "TradeStation/config/live_symbols.yaml"
        self._load_live_symbols_config(config_path)

        # ----------------------------------------------------------------- #
        # Order & Position tracking
        # ----------------------------------------------------------------- #
        self.active_orders = {}
        self.all_orders = {}
        self.bracket_children = {}
        self.positions = {}
        self.broker_positions = {}
        self.last_position_sync = None
        self.position_sync_interval = 8  # seconds
        self.last_status_check = datetime.now(timezone.utc)
        self._pending_stop_replace = {}

    def _load_live_symbols_config(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"[LIVE API] live_symbols.yaml NOT FOUND!\n"
                f"   Path: {config_path}\n"
                f"   This file is REQUIRED for margin, tick size, and commissions.\n"
                f"   Bot will not run without it."
            )

        try:
            import yaml
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            
            plat = cfg["symbols"]["platinum"]

            # Update instance variables
            self.data_symbol = plat.get("continuous", self.data_symbol)
            self.trading_symbol = plat.get("active_contract", self.trading_symbol)
            self.tick_size = float(plat.get("tick_size", self.tick_size))
            self.order_timeout = int(cfg.get("order_timeout_seconds", self.order_timeout))

            # === Build asset using instance vars (not plat) ===
            asset = {
                "symbol": self.data_symbol,           # ← @PL
                "tick_size": self.tick_size,          # ← 0.1
                "tick_value": self.tick_value,        # ← 5.0
                "initial_margin": float(plat.get("initial_margin", 8250.0)),
                "maintenance_margin": float(plat.get("maintenance_margin", 7500.0)),
                "commission_per_contract": float(plat.get("commission_per_contract", 2.0)),
                "fee_per_trade": float(plat.get("fee_per_trade", 1.5)),
            }
            self.assets = [asset]

            self.config = {
                "initial_cash": 25000,
                "commission_per_contract": asset["commission_per_contract"],
                "fee_per_trade": asset["fee_per_trade"],
            }

            print(f"[INIT] Data: {self.data_symbol} | Trading: {self.trading_symbol} | "
                  f"TickSize: {self.tick_size} | Timeout: {self.order_timeout}s | "
                  f"Margin: ${asset['initial_margin']:,}")

        except Exception as e:
            raise FileNotFoundError(
                f"[LIVE API] FAILED to parse live_symbols.yaml: {e}\n"
                f"   Path: {config_path}\n"
                f"   Fix the YAML syntax or content."
            ) from e

    # --------------------------------------------------------------------- #
    # AUTHENTICATION HELPERS
    # --------------------------------------------------------------------- #
    def _try_refresh_token(self):
        if not os.path.exists(self.config_path):
            return False
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            refresh_token = config.get("REFRESH_TOKEN")
            if not refresh_token:
                return False
            return self._refresh_access_token(refresh_token)
        except Exception as e:
            print(f"[REFRESH] Failed: {e}")
            return False

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
            'scope': 'openid MarketData ReadAccount offline_access',
        }
        auth_request_url = AUTH_URL + '?' + urllib.parse.urlencode(params)
        print("[AUTH] Open browser and login:")
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
        if not response.ok:
            print(f"[AUTH] ERROR: {response.text}")
            return None
        tokens = response.json()
        access_token = tokens.get('access_token')
        refresh_token = tokens.get('refresh_token')
        if refresh_token:
            config["REFRESH_TOKEN"] = refresh_token
            with open(self.config_path, "w") as f:
                yaml.safe_dump(config, f)
            print("[AUTH] REFRESH_TOKEN saved.")
        return access_token

    def _refresh_access_token(self, refresh_token):
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            data = {
                "grant_type": "refresh_token",
                "client_id": config["CLIENT_ID"],
                "client_secret": config["CLIENT_SECRET"],
                "refresh_token": refresh_token,
            }
            resp = requests.post("https://signin.tradestation.com/oauth/token", data=data, timeout=10)
            if resp.ok:
                tokens = resp.json()
                new_access = tokens.get("access_token")
                new_refresh = tokens.get("refresh_token")
                if new_access:
                    self.access_token = new_access
                    self.headers = {'Authorization': f"Bearer {self.access_token}"}
                    print("[TOKEN] Refreshed access token.")
                    if new_refresh and new_refresh != refresh_token:
                        config["REFRESH_TOKEN"] = new_refresh
                        with open(self.config_path, "w") as f:
                            yaml.safe_dump(config, f)
                        print("[TOKEN] Updated REFRESH_TOKEN.")
                    return True
            else:
                print(f"[TOKEN] Refresh failed: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"[TOKEN] Exception: {e}")
        return False

    # --------------------------------------------------------------------- #
    # POSITION & ORDER SYNC (FIXED + RECOVER STOPS)
    # --------------------------------------------------------------------- #
    def _sync_positions_and_orders(self):
        now = datetime.now(timezone.utc)
        print(f"[SYNC] Running position sync at {now.strftime('%H:%M:%S')}")

        # --- POSITIONS: REPLACE self.positions COMPLETELY ---
        url_pos = f"{BASE_URI}/v3/brokerage/accounts/{self.account_id}/positions"
        try:
            r = requests.get(url_pos, headers=self.headers, timeout=10)
            if r.status_code == 401 and self._try_refresh_token():
                r = requests.get(url_pos, headers=self.headers, timeout=10)
            if r.ok:
                data = r.json()
                positions = data.get("Positions", [])
                print(f"[SYNC] TradeStation reports {len(positions)} positions")

                # WIPE AND REBUILD self.positions
                self.positions = {}
                for p in positions:
                    sym = p["Symbol"]
                    long_short = p.get("LongShort", "").upper()
                    quantity = float(p.get("Quantity", 0))
                    
                    # Map LongShort + Quantity → qty
                    if long_short == "LONG":
                        qty = quantity
                    elif long_short == "SHORT":
                        qty = -quantity
                    else:
                        qty = 0
               
                    if qty == 0:
                        continue
                    avg_price = float(p.get("AveragePrice", 0)) if p.get("AveragePrice") != "0" else 0.0
                    self.positions[sym] = Position(symbol=sym, contract_size=50)
                    self.positions[sym].qty = qty
                    self.positions[sym].avg_entry_price = avg_price
                    print(f"[SYNC] LOADED {sym} qty={qty} @ {avg_price}")
        except Exception as e:
            print(f"[POS SYNC] ERROR: {e}")

        # --- RECOVER STOPS FROM OPEN ORDERS ---
        url_orders = f"{BASE_URI}/v3/brokerage/accounts/{self.account_id}/orders"
        try:
            r = requests.get(url_orders, headers=self.headers, timeout=10)
            if r.status_code == 401 and self._try_refresh_token():
                r = requests.get(url_orders, headers=self.headers, timeout=10)
            if r.ok:
                all_orders = r.json().get("Orders", [])
                for o in all_orders:
                    oid = o.get("OrderID")
                    legs = o.get("Legs", [])
                    if not legs: continue
                    sym = legs[0].get("Symbol")
                    otype = o.get("OrderType")
                    stop_price = o.get("StopPrice")
                    status_desc = o.get("StatusDescription", "Unknown")
                    if otype == "StopMarket" and sym in self.positions:
                        pos = self.positions[sym]
                        if pos.qty != 0 and status_desc == "Received":
                            pos.stop_loss_price = float(stop_price or 0)
                            pos.stop_order_id = oid
                            print(f"[SYNC] RECOVERED ACTIVE STOP {oid} @ {pos.stop_loss_price} for {sym} (Received)")
        except Exception as e:
            print(f"[ORDER SYNC] EXCEPTION: {e}")

        self.last_position_sync = now

    # --------------------------------------------------------------------- #
    # CSV INITIAL LOAD
    # --------------------------------------------------------------------- #
    def _load_and_update_csv(self):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'w') as f:
                f.write("Open,High,Low,Close,TotalVolume,TimeStamp\n")
            print(f"[CSV] Created new file with header: {CSV_PATH}")

        if os.path.exists(CSV_PATH):
            try:
                csv_df = pd.read_csv(CSV_PATH)
                csv_df["TimeStamp"] = pd.to_datetime(csv_df["TimeStamp"], utc=True)
                print(f"[CSV] Loaded {len(csv_df)} rows from {CSV_PATH}")
            except Exception as e:
                print(f"[CSV] Error reading CSV: {e}")
                csv_df = pd.DataFrame()
        else:
            csv_df = pd.DataFrame()

        now = datetime.now(timezone.utc)
        if not csv_df.empty and "TimeStamp" in csv_df.columns:
            last_ts = csv_df["TimeStamp"].max()
            if last_ts >= now:
                print(f"[CSV] Up to date. Last bar: {last_ts}")
                full_df = csv_df
            else:
                start_date = last_ts + timedelta(minutes=self.bar_interval)
                print(f"[CSV] Last timestamp: {last_ts} → downloading from {start_date}")
                new_df = self._fetch_new_bars(start_date, end_date=now)
                if not new_df.empty:
                    full_df = pd.concat([csv_df, new_df], ignore_index=True).drop_duplicates(subset="TimeStamp")
                    print(f"[CSV] Appended {len(new_df)} new bars")
                else:
                    full_df = csv_df
        else:
            start_date = now - timedelta(days=1)
            print(f"[CSV] No data. Downloading from {start_date}")
            new_df = self._fetch_new_bars(start_date, end_date=now)
            full_df = new_df if not new_df.empty else pd.DataFrame()

        if not full_df.empty:
            full_df_out = full_df.copy()
            last_ts = full_df_out["TimeStamp"].max()
            self.last_minute_written = self.round_to_bar_boundary(last_ts)
            full_df_out["TimeStamp"] = full_df_out["TimeStamp"].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            full_df_out.to_csv(CSV_PATH, index=False)
            self.df = pd.DataFrame({
                "date": full_df["TimeStamp"],
                "open": full_df["Open"].astype(float),
                "high": full_df["High"].astype(float),
                "low": full_df["Low"].astype(float),
                "close": full_df["Close"].astype(float),
                "volume": full_df["TotalVolume"].astype(int),
            })
            self.df["timestamp"] = self.df["date"].astype("int64") // 10**9
        else:
            self.last_minute_written = self.round_to_bar_boundary(now)
        print(f"[INIT] Internal df: {len(self.df)} bars")

    # --------------------------------------------------------------------- #
    # MARKET DATA HELPERS
    # --------------------------------------------------------------------- #
    def _fetch_current_equity(self):
        url = f"{BASE_URI}/v3/brokerage/accounts/{self.account_id}/balances"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 401 and self._try_refresh_token():
                resp = requests.get(url, headers=self.headers, timeout=10)
            if not resp.ok:
                return None
            bal = resp.json()["Balances"][0]
            return float(bal["Equity"])
        except Exception as e:
            print(f"[EQUITY] Error: {e}")
            return None

    def _fetch_new_bars(self, start_date, end_date=None):
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date >= end_date:
            print(f"[DOWNLOAD] Skipping: start_date {start_date} >= end_date {end_date}")
            return pd.DataFrame()
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (
            f"{BASE_URI}/v3/marketdata/barcharts/{urllib.parse.quote(self.data_symbol)}"
            f"?unit=Minute&interval={self.bar_interval}"
            f"&firstdate={start_str}&lastdate={end_str}"
        )
        print(f"[DOWNLOAD] GET {url}")
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            if resp.status_code == 401:
                if not self._try_refresh_token():
                    return pd.DataFrame()
                resp = requests.get(url, headers=self.headers, timeout=15)
            if not resp.ok:
                print(f"[DOWNLOAD] ERROR: {resp.text}")
                return pd.DataFrame()
            data = resp.json()
            bars = data.get("Bars", [])
            if not bars:
                print("[DOWNLOAD] No new bars.")
                return pd.DataFrame()
            df_data = []
            for entry in bars:
                bar = entry if "Open" in entry else entry.get("Bar", {})
                if not bar:
                    continue
                row = {
                    "Open": float(bar["Open"]),
                    "High": float(bar["High"]),
                    "Low": float(bar["Low"]),
                    "Close": float(bar["Close"]),
                    "TotalVolume": int(bar["TotalVolume"]),
                    "TimeStamp": bar.get("TimeStamp") or bar.get("BarStartTime"),
                }
                df_data.append(row)
            df = pd.DataFrame(df_data)
            df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], utc=True)
            return df
        except Exception as e:
            print(f"[DOWNLOAD] Exception: {e}")
            return pd.DataFrame()

    def _get_quote(self, symbol: str = None) -> dict:
        sym = symbol or self.trading_symbol
        url = f"{BASE_URI}/v3/marketdata/quotes/{sym}"
        try:
            r = requests.get(url, headers=self.headers, timeout=10)
            if r.status_code == 401 and self._try_refresh_token():
                r = requests.get(url, headers=self.headers, timeout=10)
            if not r.ok:
                print(f"[QUOTE] HTTP {r.status_code}: {r.text}")
                return {"Bid": 0.0, "Ask": 0.0}
            data = r.json()
            if isinstance(data, dict) and "Quotes" in data:
                quotes = data["Quotes"]
            elif isinstance(data, list):
                quotes = data
            else:
                quotes = []
            if quotes and isinstance(quotes[0], dict):
                q = quotes[0]
                bid = float(q.get("Bid", 0))
                ask = float(q.get("Ask", 0))
                return {"Bid": bid, "Ask": ask}
            else:
                print(f"[QUOTE] No valid quote for {sym}: {data}")
                return {"Bid": 0.0, "Ask": 0.0}
        except Exception as e:
            print(f"[QUOTE] Exception: {e}")
            return {"Bid": 0.0, "Ask": 0.0}
            
    def get_symbol_timeframe(self, symbol=None):
        return f"{self.bar_interval}m"  # ← MUST return "5m"

    # --------------------------------------------------------------------- #
    # APIInterface REQUIRED METHODS
    # --------------------------------------------------------------------- #
    #def connect(self):
    #    print("[LIVE] Starting SMART POLLING (official 5-min bars every 20s)...")
    #    self.running = True
    #    self.thread = threading.Thread(target=self._poll_loop, daemon=True)
    #    self.thread.start()
    
    def connect(self):
        """Start authentication, load CSV, sync positions, and begin polling"""
        if not self.access_token:
            if self._try_refresh_token():
                print("[INIT] Token refreshed from REFRESH_TOKEN.")
            else:
                print("[INIT] Using browser login...")
                self.access_token = self._authenticate_and_get_token()
        if not self.access_token:
            raise RuntimeError("Failed to obtain TradeStation access_token.")
        self.headers = {'Authorization': f"Bearer {self.access_token}"}
        
        self._load_and_update_csv()
        print(f"[INIT] Final dataset: {len(self.df)} bars. Starting live polling...")
        
        self._sync_positions_and_orders()
        
        # Start polling thread
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def disconnect(self):
        print("[LIVE] Stopping...")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def get_historical_data(self, asset, timeframe, start, end):
        """Not used in live mode – raise."""
        raise NotImplementedError("Historical data via CSV only in live mode")

    # --------------------------------------------------------------------- #
    # STOP-LOSS MODIFICATION
    # --------------------------------------------------------------------- #
    def modify_stop_loss(self, symbol: str, new_price: float) -> bool:
        pos = self.positions.get(symbol)
        if not pos or pos.qty == 0:
            return False
        new_price = round_to_tick(new_price, self.tick_size)
        cur_price = pos.stop_loss_price or 0.0

        # Determine direction
        tighter = (pos.qty > 0 and new_price > cur_price) or (pos.qty < 0 and new_price < cur_price)
        if not tighter:
            return False

        # --- CASE 1: STOP EXISTS → MODIFY ---
        if pos.stop_order_id:
            payload = {"StopPrice": f"{new_price:.2f}"}
            url = f"{BASE_URI}/v3/orderexecution/orders/{pos.stop_order_id}"
            r = requests.put(url, json=payload, headers=self.headers, timeout=10)
            if r.status_code == 401 and self._try_refresh_token():
                r = requests.put(url, json=payload, headers=self.headers, timeout=10)
            if r.ok:
                pos.stop_loss_price = new_price
                print(f"[STOP] MODIFIED {pos.stop_order_id} → {new_price:.2f}")
                return True
            else:
                print(f"[STOP] MODIFY FAILED: {r.text}")
                return False

        # --- CASE 2: NO STOP → CREATE BRACKET ---
        side = "SELL" if pos.qty > 0 else "BUY"
        payload = {
            "AccountID": self.account_id,
            "Symbol": self.trading_symbol,
            "Quantity": str(int(abs(pos.qty))),
            "OrderType": "StopMarket",
            "TradeAction": side,
            "StopPrice": f"{new_price:.2f}",
            "TimeInForce": {"Duration": "DAY"},
        }
        url = f"{BASE_URI}/v3/orderexecution/orders"
        r = requests.post(url, json=payload, headers=self.headers, timeout=10)
        if r.status_code == 401 and self._try_refresh_token():
            r = requests.post(url, json=payload, headers=self.headers, timeout=10)
        if r.ok:
            stop_id = r.json()["Orders"][0]["OrderID"]
            pos.stop_order_id = stop_id
            pos.stop_loss_price = new_price
            print(f"[STOP] CREATED NEW STOP {stop_id} @ {new_price:.2f}")
            return True
        else:
            print(f"[STOP] CREATE FAILED: {r.text}")
            return False

    # --------------------------------------------------------------------- #
    # ORDER PLACEMENT (WITH AUTO-TIGHTEN TEST)
    # --------------------------------------------------------------------- #
    def place_order(self, order_dict: dict) -> dict:
        symbol = self.trading_symbol
        side = order_dict["side"].upper()
        qty = int(order_dict["qty"])
        order_type = order_dict.get("order_type", "market").lower()
        
        print("[DEBUG] Order Placed")
        print(order_dict)

        if order_type == "market":
            quote = self._get_quote(self.trading_symbol)
            bid, ask = quote.get("Bid", 0), quote.get("Ask", 0)
            if bid <= 0 or ask <= 0:
                last = self.df["close"].iloc[-1] if not self.df.empty else 1000.0
                bid = ask = last
            entry_price = ask + self.tick_size if side == "BUY" else bid - self.tick_size
            entry_price = round_to_tick(entry_price, self.tick_size)
            entry_type = "Limit"
        else:
            entry_price = float(order_dict.get("price"))
            entry_price = round_to_tick(entry_price, self.tick_size)
            entry_type = "Limit"

        parent_payload = {
            "AccountID": self.account_id,
            "Symbol": symbol,
            "Quantity": str(int(abs(qty))),
            "OrderType": entry_type,
            "TradeAction": side,
            "LimitPrice": f"{entry_price:.2f}",
            "TimeInForce": {"Duration": "DAY"},
        }

        offset = order_dict.get("original_stop_offset")
        stop_price = None
        if offset is not None:
            offset = float(offset)
            stop_price = entry_price - offset if side == "BUY" else entry_price + offset
            stop_price = round_to_tick(stop_price, self.tick_size)

        url = f"{BASE_URI}/v3/orderexecution/orders"
        print(f"[DEBUG] Sending ENTRY:\n{json.dumps(parent_payload, indent=2)}")
        try:
            resp = requests.post(url, json=parent_payload, headers=self.headers, timeout=15)
            if resp.status_code == 401 and self._try_refresh_token():
                resp = requests.post(url, json=parent_payload, headers=self.headers, timeout=15)
            if not resp.ok:
                print(f"[ORDER] ERROR {resp.status_code}: {resp.text}")
                order_dict["status"] = "rejected"
                order_dict["error"] = resp.text
                return order_dict

            parent_id = resp.json()["Orders"][0]["OrderID"]
            print(f"[ORDER] ENTRY SENT → ParentID={parent_id}")
            self.active_orders[parent_id] = {
                "submit_time": datetime.now(timezone.utc),
                "side": side.lower(),
                "type": "parent"
            }
            self.all_orders[parent_id] = self.active_orders[parent_id].copy()

            if stop_price is not None:
                stop_payload = {
                    "AccountID": self.account_id,
                    "Symbol": symbol,
                    "Quantity": str(int(abs(qty))),
                    "OrderType": "StopMarket",
                    "TradeAction": "SELL" if side == "BUY" else "BUY",
                    "StopPrice": f"{stop_price:.2f}",
                    "TimeInForce": {"Duration": "DAY"},
                    "ParentOrderID": parent_id
                }
                print(f"[STOP] ATTACHING StopMarket @ {stop_price:.2f}")
                stop_resp = requests.post(url, json=stop_payload, headers=self.headers, timeout=10)
                if stop_resp.status_code == 401 and self._try_refresh_token():
                    stop_resp = requests.post(url, json=stop_payload, headers=self.headers, timeout=10)

                if stop_resp.ok:
                    stop_id = stop_resp.json()["Orders"][0]["OrderID"]
                    print(f"[STOP] ATTACHED → ID {stop_id}")
                    self.bracket_children[parent_id] = [stop_id]

                    if symbol not in self.positions:
                        self.positions[symbol] = Position(symbol=symbol, contract_size=50)
                    pos = self.positions[symbol]
                    pos.stop_loss_price = stop_price
                    pos.stop_order_id = stop_id
                    print(f"[POS] Stop ID saved: {stop_id} @ {stop_price}")

                else:
                    print(f"[STOP] FAILED {stop_resp.status_code}: {stop_resp.text}")

            order_dict.update({
                "order_id": parent_id,
                "status": "Sent",
                "bracket_children": self.bracket_children.get(parent_id, [])
            })
            return order_dict

        except Exception as e:
            print(f"[ORDER] EXCEPTION: {e}")
            order_dict["status"] = "failed"
            return order_dict
            
    def _cancel_stop_if_exists(self, symbol: str):
        """Cancel stop order if position is flat."""
        pos = self.positions.get(symbol)
        if not pos:
            return
        if pos.stop_order_id and pos.stop_order_id != "UNKNOWN":
            print(f"[CANCEL STOP] Cancelling stop {pos.stop_order_id}")
            self.cancel_order(pos.stop_order_id)
            pos.stop_order_id = None
            pos.stop_loss_price = None

    # --------------------------------------------------------------------- #
    # ORDER STATUS / CANCELLATION
    # --------------------------------------------------------------------- #
    def get_order_status(self, order_id: str) -> dict:
        if order_id == "UNKNOWN":
            return {"status": "unknown"}
        url = f"{BASE_URI}/v3/orderexecution/orders/{order_id}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 401 and self._try_refresh_token():
                resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code in [404, 405]:
                return {"status": "closed_by_405"}
            if not resp.ok:
                return {"status": "error"}
            result = resp.json()
            return {"status": result.get("Order", {}).get("Status", "unknown")}
        except:
            return {"status": "error"}

    def cancel_order(self, order_id):
        if order_id == "UNKNOWN":
            return
        url = f"{BASE_URI}/v3/orderexecution/orders/{order_id}"
        try:
            resp = requests.delete(url, headers=self.headers, timeout=10)
            if resp.status_code == 401 and self._try_refresh_token():
                resp = requests.delete(url, headers=self.headers, timeout=10)
            if resp.ok:
                print(f"[CANCEL] SUCCESS: {order_id}")
                self.active_orders.pop(order_id, None)
            else:
                print(f"[CANCEL] FAILED {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[CANCEL] EXCEPTION: {e}")

    def _check_order_timeouts(self, now):
        to_cancel = []
        for order_id, info in self.active_orders.items():
            elapsed = (now - info["submit_time"]).total_seconds()
            if elapsed > self.order_timeout:
                print(f"[TIMEOUT] Cancelling {order_id} after {elapsed:.1f}s")
                to_cancel.append(order_id)
        for order_id in to_cancel:
            self.cancel_order(order_id)

    def _check_active_orders_status(self):
        if not self.all_orders:
            return
        to_remove = []
        for order_id in list(self.all_orders.keys()):
            try:
                status_info = self.get_order_status(order_id)
                status = status_info.get("status", "unknown")
                if status in ["Filled", "Cancelled", "Rejected", "closed_by_405"]:
                    to_remove.append(order_id)
            except:
                to_remove.append(order_id)
        for oid in to_remove:
            self.active_orders.pop(oid, None)
            self.all_orders.pop(oid, None)

    # --------------------------------------------------------------------- #
    # FILL DETECTION + POSITION UPDATE
    # --------------------------------------------------------------------- #
    def _update_position_on_fill(self, order_id: str, fill_qty: float, fill_price: float, side: str):
        """Called when a parent order is filled – update Position object."""
        sym = self.trading_symbol
        if sym not in self.positions:
            self.positions[sym] = Position(symbol=sym, contract_size=50)
        pos = self.positions[sym]

        # Convert side to signed qty
        signed_qty = fill_qty if side.lower() == "buy" else -fill_qty
        realized = pos.update_on_fill(
            fill_price=fill_price,
            fill_qty=fill_qty,
            side=side.lower(),
            tick_size=self.tick_size,
            tick_value=self.tick_value,
        )
        print(f"[FILL] {sym} {side.upper()} {fill_qty} @ {fill_price:.2f} → PnL {realized:+.2f}")
        # Emit equity update
        equity = self._fetch_current_equity() or 25000.0
        self.equity_updated.emit(equity, self._last_bar_key)
        self._write_equity_to_csv(self._last_bar_key, equity)

    def _check_active_orders_status(self):
        """Detect fills and update positions."""
        if not self.all_orders:
            return
        to_remove = []
        for order_id in list(self.all_orders.keys()):
            status_info = self.get_order_status(order_id)
            status = status_info.get("status", "unknown")
            if status in ["Filled", "Cancelled", "Rejected", "closed_by_405"]:
                # --- FETCH FILL DETAILS ---
                url = f"{BASE_URI}/v3/orderexecution/orders/{order_id}"
                try:
                    r = requests.get(url, headers=self.headers, timeout=10)
                    if r.ok:
                        o = r.json().get("Order", {})
                        filled_qty = float(o.get("FilledQuantity", 0))
                        avg_fill = float(o.get("AverageFillPrice", 0))
                        side = o.get("TradeAction", "").upper()
                        if filled_qty > 0 and avg_fill > 0:
                            self._update_position_on_fill(order_id, filled_qty, avg_fill, side)
                except Exception as e:
                    print(f"[FILL CHECK] Error: {e}")
                to_remove.append(order_id)
        for oid in to_remove:
            self.active_orders.pop(oid, None)
            self.all_orders.pop(oid, None)
            self.bracket_children.pop(oid, None)

    # --------------------------------------------------------------------- #
    # LIVE POLLING LOOP
    # --------------------------------------------------------------------- #
    def _poll_loop(self):
        print("[LIVE] Poll loop running.")
        last_timeout_check = datetime.now(timezone.utc)
        last_status_check = datetime.now(timezone.utc)
        while self.running:
            now = datetime.now(timezone.utc)

            if (now - last_timeout_check).total_seconds() >= 5:
                self._check_order_timeouts(now)
                last_timeout_check = now

            if (now - last_status_check).total_seconds() >= 2:
                self._check_active_orders_status()
                last_status_check = now

            self._sync_positions_and_orders()

            # TIGHTEN EVERY 20s
            #print(f"[POLL] Positions: {[(sym, pos.qty, pos.stop_loss_price, pos.stop_order_id) for sym, pos in self.positions.items()]}")
            #for sym, pos in self.positions.items():
            #    if pos.qty > 0 and pos.stop_order_id and pos.stop_loss_price:
            #        new_stop = round_to_tick(pos.stop_loss_price + 10 * self.tick_size, self.tick_size)
            #        print(f"[POLL TIGHTEN] {sym} qty={pos.qty} | Stop {pos.stop_loss_price} → {new_stop}")
            #        if self.modify_stop_loss(sym, new_stop):
            #            print(f"[POLL TIGHTEN] SUCCESS: Stop now at {new_stop:.2f}")

            equity = self._fetch_current_equity()
            if equity is not None:
                self.equity_updated.emit(equity, self._last_bar_key)
                self._write_equity_to_csv(self._last_bar_key, equity)

            try:
                new_bars = self._fetch_latest_official_bars()
                if new_bars is not None and not new_bars.empty:
                    self._merge_official_bars(new_bars)
            except Exception as e:
                print(f"[POLL] Error: {e}")

            time.sleep(self.poll_interval)

    def _fetch_latest_official_bars(self):
        url = (
            f"{BASE_URI}/v3/marketdata/barcharts/{urllib.parse.quote(self.data_symbol)}"
            f"?interval={self.bar_interval}&unit=Minute&barsback=2"
        )
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code == 401:
                if not self._try_refresh_token():
                    return None
                resp = requests.get(url, headers=self.headers, timeout=10)
            if not resp.ok:
                print(f"[POLL] HTTP {resp.status_code}: {resp.text}")
                return None
            data = resp.json().get("Bars", [])
            if not data:
                return pd.DataFrame()
            rows = []
            for bar in data:
                b = bar if "Open" in bar else bar.get("Bar", {})
                if not b:
                    continue
                rows.append({
                    "Open": float(b["Open"]),
                    "High": float(b["High"]),
                    "Low": float(b["Low"]),
                    "Close": float(b["Close"]),
                    "TotalVolume": int(b["TotalVolume"]),
                    "TimeStamp": pd.to_datetime(b.get("TimeStamp") or b.get("BarStartTime"), utc=True)
                })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[POLL] Exception: {e}")
            return None

    def _merge_official_bars(self, new_df):
        if new_df.empty or self.last_minute_written is None:
            return
        for _, row in new_df.iterrows():
            ts = row["TimeStamp"]
            minute = self.round_to_bar_boundary(ts)
            if minute > self.last_minute_written:
                new_bar = {
                    'date': ts,
                    'open': row["Open"],
                    'high': row["High"],
                    'low': row["Low"],
                    'close': row["Close"],
                    'volume': row["TotalVolume"],
                    'timestamp': int(ts.timestamp())
                }
                self.df = pd.concat([self.df, pd.DataFrame([new_bar])], ignore_index=True)
                self.last_minute_written = minute
                self._append_bar_to_csv(new_bar)
                print(f"[BAR] NEW: {ts.strftime('%H:%M')} OPEN={row['Open']:.2f}")
            elif minute == self.last_minute_written:
                idx = self.df.index[-1]
                changed = False
                if row["High"] > self.df.at[idx, 'high']:
                    self.df.at[idx, 'high'] = row["High"]
                    changed = True
                if row["Low"] < self.df.at[idx, 'low']:
                    self.df.at[idx, 'low'] = row["Low"]
                    changed = True
                if abs(row["Close"] - self.df.at[idx, 'close']) > 1e-6:
                    self.df.at[idx, 'close'] = row["Close"]
                    changed = True
                if row["TotalVolume"] != self.df.at[idx, 'volume']:
                    self.df.at[idx, 'volume'] = row["TotalVolume"]
                    changed = True
                if changed:
                    self._update_csv_last_line()
                    print(f"[BAR] UPD: {ts.strftime('%H:%M')} H={row['High']:.2f} L={row['Low']:.2f} C={row['Close']:.2f}", end='\r')
        self.bar_updated.emit(self.df.copy())

    # --------------------------------------------------------------------- #
    # CSV HELPERS
    # --------------------------------------------------------------------- #
    def _append_bar_to_csv(self, bar):
        line = f"{bar['open']:.2f},{bar['high']:.2f},{bar['low']:.2f},{bar['close']:.2f},{int(bar['volume'])},{bar['date'].strftime('%Y-%m-%dT%H:%M:%S.%fZ')}\n"
        with open(CSV_PATH, "a") as f:
            f.write(line)

    def _update_csv_last_line(self):
        if len(self.df) <= 1:
            return
        row = self.df.iloc[-1]
        line = f"{row['open']:.2f},{row['high']:.2f},{row['low']:.2f},{row['close']:.2f},{int(row['volume'])},{pd.Timestamp(row['date']).strftime('%Y-%m-%dT%H:%M:%S.%fZ')}\n"
        try:
            with open(CSV_PATH, 'r+') as f:
                lines = f.readlines()
                if lines:
                    lines[-1] = line
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()
        except Exception as e:
            print(f"[CSV] Failed to update last line: {e}")

    def round_to_bar_boundary(self, ts):
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            dt = ts.tz_convert('UTC').tz_localize(None)
        else:
            dt = pd.Timestamp(ts)
        dt = dt.floor('min')
        total_minutes = dt.hour * 60 + dt.minute
        bar_minute = (total_minutes // self.bar_interval) * self.bar_interval
        return dt.replace(
            hour=bar_minute // 60,
            minute=bar_minute % 60,
            second=0, microsecond=0
        )

    def _write_equity_to_csv(self, bar_key: str, equity: float):
        line = f"{bar_key},{equity:.2f}\n"
        lines = []
        if os.path.exists(self.equity_file):
            with open(self.equity_file, 'r') as f:
                lines = f.readlines()
        found = False
        for i, existing_line in enumerate(lines):
            if existing_line.startswith(bar_key + ","):
                lines[i] = line
                found = True
                break
        if not found:
            lines.append(line)
        with open(self.equity_file, 'w') as f:
            f.writelines(lines)

    # --------------------------------------------------------------------- #
    # PASSTHROUGH APIInterface METHODS
    # --------------------------------------------------------------------- #
    def get_positions(self):
        """
        Return a dict matching BaseStrategyBot's expectations:
        {
            "SYMBOL": {
                "qty": int,
                "avg_entry_price": float,
                "realized_pnl": float,
                "unrealized_pnl": float,
                "stop_loss_price": float | None,
                "take_profit": None
            }
        }
        """
        current_price = self.df["close"].iloc[-1] if not self.df.empty else 0.0
        result = {}
        for sym, p in self.positions.items():
            result[sym] = {
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "realized_pnl": p.realized_pnl,
                "unrealized_pnl": p.get_unrealized_pnl(
                    current_price, self.tick_size, self.tick_value
                ),
                "stop_loss_price": p.stop_loss_price,
                "take_profit": p.take_profit,
            }
        return result

    def get_total_pnl(self):
        return sum(p.realized_pnl for p in self.positions.values())

    def get_portfolio(self):
        equity = self._fetch_current_equity() or 25000.0
        # Rough margin estimate: sum of |qty| * initial_margin (fallback 2500 per contract)
        used = sum(abs(p.qty) * 2500 for p in self.positions.values())
        available = max(0.0, equity - used)
        return {
            "cash": equity,
            "total_equity": equity,
            "available_equity": available,
            "positions": self.get_positions(),
            "open_orders": list(self.active_orders.keys()),
        }

    def get_asset_list(self):
        return [self.data_symbol]

    def get_latest_data(self, symbol=None, window_size=None):
        _ = window_size
        return self.df

    def get_asset_data(self, symbol=None):
        return self.df

    @property
    def _last_bar_key(self):
        if self.df.empty:
            return "unknown"
        ts = self.df.iloc[-1]["date"]
        return ts.strftime("%Y-%m-%dT%H:%M")

    def _calc_equity(self):
        equity = self._fetch_current_equity()
        return equity if equity is not None else 25000.0
