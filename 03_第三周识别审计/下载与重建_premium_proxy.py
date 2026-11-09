# -*- coding: utf-8 -*-
"""Download Binance premium-index minute data and construct a transparent proxy.

The proxy is explicitly not historical predicted funding. It uses the trailing
8-hour mean premium plus a constant 8-hour interest component and clamps it to
the current documented cap/floor. Historical rule changes still require a
separate rule-version table before a final paper specification.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://fapi.binance.com"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
LIMIT = 1_500
MINUTE_MS = 60_000
INTERVAL_MINUTES = 8 * 60
INTEREST_8H = 0.0001
CAPS = {"BTCUSDT": 0.00375, "ETHUSDT": 0.0075}
FLOORS = {"BTCUSDT": -0.00375, "ETHUSDT": -0.0075}

ROOT = Path(__file__).resolve().parent
WEEK2 = ROOT.parent / "02_第二周机制数据管道"
DATA = ROOT / "data"


def get_json(session: requests.Session, params: dict) -> list:
    for attempt in range(6):
        response = session.get(BASE_URL + "/fapi/v1/premiumIndexKlines", params=params, timeout=30)
        if response.status_code == 429:
            wait = float(response.headers.get("Retry-After", min(60, 5 * (attempt + 1))))
            print(f"429 rate limit; sleeping {wait:.1f}s")
            time.sleep(wait)
            continue
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("code", 0) < 0:
            raise RuntimeError(payload)
        return payload
    raise RuntimeError("premium-index endpoint rate-limit retries exhausted")


def download_symbol(session: requests.Session, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        payload = get_json(session, {"symbol": symbol, "interval": "1m", "startTime": cursor, "endTime": end_ms, "limit": LIMIT})
        if not payload:
            break
        rows.extend(payload)
        last = int(payload[-1][0])
        if last < cursor:
            break
        cursor = last + MINUTE_MS
        if len(rows) % 30_000 == 0:
            print(symbol, len(rows), "premium rows")
        time.sleep(0.25)
    if not rows:
        raise RuntimeError(f"no premium rows for {symbol}")
    frame = pd.DataFrame(rows).rename(columns={0: "timestamp", 1: "open", 2: "high", 3: "low", 4: "close"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["symbol"] = symbol
    return frame[["timestamp", "symbol", "open", "high", "low", "close"]].drop_duplicates(["timestamp", "symbol"]).sort_values("timestamp")


def reconstruct(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for symbol, part in frame.groupby("symbol", sort=False):
        part = part.sort_values("timestamp").copy().set_index("timestamp")
        # Use only information available up to t. The realized rate at the
        # settlement timestamp is excluded from the rolling mean by shift(1).
        trailing = part["close"].shift(1).rolling(INTERVAL_MINUTES, min_periods=INTERVAL_MINUTES).mean()
        raw = trailing + INTEREST_8H
        part["premium_trailing_8h"] = trailing
        part["reconstructed_pre_funding_proxy"] = raw.clip(FLOORS[symbol], CAPS[symbol])
        part["proxy_cap"] = CAPS[symbol]
        part["proxy_floor"] = FLOORS[symbol]
        output.append(part.reset_index())
    return pd.concat(output, ignore_index=True)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    basis = pd.read_csv(WEEK2 / "data" / "perp_spot_index_basis_1m.csv", usecols=["timestamp"])
    ts = pd.to_datetime(basis["timestamp"], utc=True, format="mixed")
    end = ts.max().ceil("min")
    start = max(ts.min(), end - pd.Timedelta(days=45))
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    session = requests.Session()
    session.headers.update({"User-Agent": "funding-settlement-week3/1.0"})
    frames = []
    for symbol in SYMBOLS:
        print("Downloading premium index", symbol, start.isoformat(), end.isoformat())
        frames.append(download_symbol(session, symbol, start_ms, end_ms))
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(DATA / "premium_index_1m.csv", index=False, encoding="utf-8-sig")
    proxy = reconstruct(raw)
    proxy.to_csv(DATA / "premium_proxy_1m.csv", index=False, encoding="utf-8-sig")
    print("saved raw rows", len(raw), "proxy rows", len(proxy))


if __name__ == "__main__":
    main()
