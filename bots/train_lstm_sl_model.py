"""
Train LSTM to select the ATR multiple (discrete action) from offline training CSVs.

Usage:
    PYTHONPATH=. python3 bots/train_lstm_sl_model.py \
        --input_dir bots/data/yahoo_finance/training_data \
        --output_dir bots/models/LSTM_Trailing_Stop_Loss \
        --lookback 32 --epochs 3 --hidden 48 --layers 1 --dropout 0.0
"""

import os, re, argparse, random, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SL_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0]
REQ = ["atr", "rsi_14", "close", "ema_21", "position_type"]  # required features

# --------- simple indicator helpers (no external deps) ---------

def _ema(values: pd.Series, span: int) -> pd.Series:
    return values.ewm(span=span, adjust=False).mean()

def _rsi14(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder's smoothing
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _atr14(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder smoothing for ATR
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

# Column aliasing: try to rename to our canonical names
ALIASES = {
    "rsi_14": ["rsi", "RSI", "rsi14", "RSI14", "rsi_14"],
    "atr": ["atr", "ATR", "atr14", "ATR14", "atr_14"],
    "ema_21": ["ema_21", "EMA_21", "ema21", "EMA21", "ema"],
    "close": ["close", "Close", "CLOSE", "adj_close", "Adj Close"],
    "high": ["high", "High", "HIGH"],
    "low":  ["low", "Low", "LOW"],
    "open": ["open", "Open", "OPEN"],
    "position_type": ["position_type", "pos_type", "posflag", "position_flag"],
    "best_sl_multiplier": ["best_sl_multiplier", "label", "y"],
}

def _rename_by_alias(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    lower_map = {c.lower(): c for c in renamed.columns}
    def pick(name, options):
        for opt in options:
            # exact
            if opt in renamed.columns:
                return opt
            # case-insensitive
            if opt.lower() in lower_map:
                return lower_map[opt.lower()]
        return None
    # Build mapping to canonical names
    mapping = {}
    for canonical, opts in ALIASES.items():
        found = pick(canonical, opts)
        if found and found != canonical:
            mapping[found] = canonical
    if mapping:
        renamed = renamed.rename(columns=mapping)
    return renamed

def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist; compute from OHLC if needed.
    """
    df = _rename_by_alias(df)

    # Close is essential
    if "close" not in df.columns:
        raise ValueError("Missing 'close' (or suitable alias).")

    # EMA_21
    if "ema_21" not in df.columns:
        df["ema_21"] = _ema(df["close"].astype(float), span=21)

    # RSI_14
    if "rsi_14" not in df.columns:
        df["rsi_14"] = _rsi14(df["close"].astype(float), period=14)

    # ATR (needs H/L; if missing, fall back to abs returns * sqrt(252) style proxy)
    if "atr" not in df.columns:
        if "high" in df.columns and "low" in df.columns:
            df["atr"] = _atr14(
                df["high"].astype(float),
                df["low"].astype(float),
                df["close"].astype(float),
                period=14
            )
        else:
            # proxy ATR from close-only (rough fallback; still useful)
            ret = df["close"].astype(float).pct_change().abs()
            proxy = (ret.rolling(14).mean() * df["close"].astype(float).shift(1)).fillna(0.0)
            df["atr"] = proxy

    # position_type: if missing, assume 0 (flat) — your generator should set +1/-1 for labels,
    # but we defensively fill with 0 so training still proceeds.
    if "position_type" not in df.columns:
        df["position_type"] = 0.0

    return df

# --------- data / model ---------

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def time_series_split(df: pd.DataFrame, train=0.7, val=0.15):
    n = len(df)
    i1 = int(n * train)
    i2 = int(n * (train + val))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int):
        self.lookback = int(lookback)
        df = _ensure_features(df)
        need = REQ + ["best_sl_multiplier"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns after enrichment: {missing}")
        # map label to class index
        map_k = {m:i for i,m in enumerate(SL_MULTIPLIERS)}
        df = df.dropna(subset=need).copy()
        df["label"] = df["best_sl_multiplier"].astype(float).map(map_k)
        df = df.dropna(subset=["label"])
        self.X, self.y = self._seqify(df)

    def _seqify(self, df: pd.DataFrame):
        X, y = [], []
        vals = df[REQ + ["label"]].to_numpy(dtype=np.float32)
        for i in range(self.lookback, len(vals)):
            win = vals[i-self.lookback:i]
            feats = win[:, :-1]
            mu = feats.mean(axis=0, keepdims=True)
            sd = feats.std(axis=0, keepdims=True) + 1e-8
            feats = (feats - mu) / sd
            X.append(feats)
            y.append(int(vals[i, -1]))
        if not X:
            return np.zeros((0, self.lookback, len(REQ)), np.float32), np.zeros((0,), np.int64)
        return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LSTMSelector(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, n_layers: int = 1, dropout: float = 0.0, n_classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=n_layers,
                            batch_first=True, dropout=(dropout if n_layers > 1 else 0.0))
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        out, _ = self.lstm(x)         # (B,T,H)
        last = out[:, -1, :]
        return self.head(last)        # (B,C)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def train_symbol(csv_path: Path, out_dir: Path, lookback: int, epochs: int, batch: int, lr: float,
                 hidden: int, layers: int, dropout: float, device: str, patience: int = 5):
    df = pd.read_csv(csv_path)
    # try to enrich/compute missing features
    try:
        df = _ensure_features(df)
    except Exception as e:
        print(f"[SKIP] {csv_path.name}: {e}")
        return False

    # requirement check
    needed = REQ + ["best_sl_multiplier"]
    if any(c not in df.columns for c in needed) or len(df) < lookback + 50:
        print(f"[SKIP] {csv_path.name}: insufficient data after enrichment")
        return False

    tr, va, te = time_series_split(df, 0.7, 0.15)
    try:
        dtr, dva, dte = SeqDataset(tr, lookback), SeqDataset(va, lookback), SeqDataset(te, lookback)
    except Exception as e:
        print(f"[SKIP] {csv_path.name}: {e}")
        return False
    if min(len(dtr), len(dva), len(dte)) == 0:
        print(f"[SKIP] {csv_path.name}: empty splits after sequencing")
        return False

    ltr = DataLoader(dtr, batch_size=batch, shuffle=True, drop_last=True)
    lva = DataLoader(dva, batch_size=batch, shuffle=False)
    lte = DataLoader(dte, batch_size=batch, shuffle=False)

    model = LSTMSelector(n_features=len(REQ), hidden=hidden, n_layers=layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_va, best_state, no_improve = -1.0, None, 0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in ltr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            va_acc = np.mean([accuracy(model(xb.to(device)), yb.to(device)) for xb, yb in lva])
        if va_acc > best_va:
            best_va, best_state, no_improve = va_acc, {k:v.cpu() for k,v in model.state_dict().items()}, 0
        else:
            no_improve += 1

        print(f"[{csv_path.name}] epoch {ep}/{epochs}  val_acc={va_acc:.3f}  best={best_va:.3f}")
        if no_improve >= patience:
            print(f"[{csv_path.name}] Early stop after {ep} epochs.")
            break

    if best_state is None: best_state = {k:v.cpu() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        te_acc = np.mean([accuracy(model(xb.to(device)), yb.to(device)) for xb, yb in lte])
    print(f"[{csv_path.name}] TEST acc={te_acc:.3f} (lookback={lookback})")

    # Export TorchScript + meta
    example = torch.randn(1, lookback, len(REQ)).to(device)
    traced = torch.jit.trace(model, example)
    m = re.match(r'^rl_stop_loss_training_(.+)\.csv$', csv_path.name)
    symbol = m.group(1) if m else csv_path.stem.replace('rl_stop_loss_training_', '')
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_path = out_dir / f"lstm_stop_loss_selector_{symbol}.pt"
    traced.save(str(ts_path))
    meta = {"lookback": lookback, "features": REQ, "sl_multipliers": SL_MULTIPLIERS}
    with open(out_dir / f"lstm_stop_loss_selector_{symbol}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVED] {ts_path}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lookback", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(in_dir.glob("rl_stop_loss_training_*.csv"))
    if not files:
        print(f"No training files in {in_dir}")
        return
    for f in files:
        try:
            train_symbol(f, out_dir, args.lookback, args.epochs, args.batch, args.lr,
                         args.hidden, args.layers, args.dropout, args.device, args.patience)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

if __name__ == '__main__':
    main()

