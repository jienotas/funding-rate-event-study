# -*- coding: utf-8 -*-
"""Download second-week mechanism data for the existing 180-day sample.

The script adds spot, index, and mark-price minute series and records the
current premium-index snapshot and exchange rules. Historical funding history
is realized funding; the current snapshot is not treated as historical predicted
funding.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://fapi.binance.com"
SPOT_URL = "https://api.binance.com"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
LIMIT = 1_500
MINUTE_MS = 60_000

ROOT = Path(__file__).resolve().parent
WEEK1 = ROOT.parent / "01_第一周可行性测试"
DATA = ROOT / "data"


def get_json(session: requests.Session, base: str, path: str, params: dict):
    for attempt in range(7):
        response = session.get(base + path, params=params, timeout=30)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after else min(60.0, 5.0 * (attempt + 1))
            print(f"429 rate limit for {path}; sleeping {wait_seconds:.1f}s")
            time.sleep(wait_seconds)
            continue
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("code", 0) < 0:
            raise RuntimeError(f"API error: {payload}")
        return payload
    raise RuntimeError(f"Too many rate-limit retries for {path}")


def load_window() -> tuple[int, int]:
    prices = pd.read_csv(WEEK1 / "data" / "perp_1m.csv", usecols=["timestamp"])
    ts = pd.to_datetime(prices["timestamp"], utc=True, format="mixed")
    # Use a bounded recent slice for the second-week mechanism test. The index
    # endpoint is rate limited; expand only after this slice passes QA.
    end = ts.max()
    start = max(ts.min(), end - pd.Timedelta(days=45))
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def klines(session: requests.Session, symbol: str, start_ms: int, end_ms: int, kind: str) -> pd.DataFrame:
    if kind == "spot":
        base, path, params_base = SPOT_URL, "/api/v3/klines", {"symbol": symbol}
    elif kind == "index":
        base, path, params_base = BASE_URL, "/fapi/v1/indexPriceKlines", {"pair": symbol}
    elif kind == "mark":
        base, path, params_base = BASE_URL, "/fapi/v1/markPriceKlines", {"symbol": symbol}
    else:
        raise ValueError(kind)

    rows = []
    cursor = start_ms
    request_count = 0
    while cursor < end_ms:
        params = {**params_base, "interval": "1m", "startTime": cursor, "endTime": end_ms, "limit": LIMIT}
        payload = get_json(session, base, path, params)
        request_count += 1
        if not payload:
            break
        rows.extend(payload)
        last = int(payload[-1][0])
        if last < cursor:
            break
        cursor = last + MINUTE_MS
        if request_count % 30 == 0:
            print(f"{kind} {symbol}: {len(rows):,} rows")
        # Index and mark endpoints are more aggressively rate limited than spot.
        time.sleep(0.8)

    if not rows:
        raise RuntimeError(f"No {kind} rows for {symbol}")
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={0: "timestamp", 4: "price"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True)
    frame["symbol"] = symbol
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    return frame[["timestamp", "symbol", "price"]].drop_duplicates(["timestamp", "symbol"]).sort_values("timestamp")


def save_api_audit(session: requests.Session) -> None:
    snapshot = get_json(session, BASE_URL, "/fapi/v1/premiumIndex", {})
    snap = pd.DataFrame(snapshot)
    if not snap.empty:
        snap = snap.loc[snap["symbol"].isin(SYMBOLS)]
        snap.to_csv(DATA / "premium_index_snapshot.csv", index=False, encoding="utf-8-sig")

    info = get_json(session, BASE_URL, "/fapi/v1/exchangeInfo", {})
    rows = []
    for item in info.get("symbols", []):
        if item.get("symbol") in SYMBOLS:
            rows.append(
                {
                    "symbol": item.get("symbol"),
                    "status": item.get("status"),
                    "contractType": item.get("contractType"),
                    "quoteAsset": item.get("quoteAsset"),
                    "marginAsset": item.get("marginAsset"),
                    "onboardDate": item.get("onboardDate"),
                    "deliveryDate": item.get("deliveryDate"),
                }
            )
    pd.DataFrame(rows).to_csv(DATA / "exchange_rules.csv", index=False, encoding="utf-8-sig")

    audit = pd.DataFrame(
        [
            {"item": "historical_realized_funding", "endpoint": "/fapi/v1/fundingRate", "status": "available", "interpretation": "历史实现资金费率"},
            {"item": "current_premium_snapshot", "endpoint": "/fapi/v1/premiumIndex", "status": "available", "interpretation": "当前快照含 lastFundingRate/nextFundingTime，不是历史预测序列"},
            {"item": "historical_predicted_funding", "endpoint": "public historical endpoint not identified", "status": "not_verified", "interpretation": "不能用 realized funding 替代"},
            {"item": "spot_1m", "endpoint": "https://api.binance.com/api/v3/klines", "status": "downloaded", "interpretation": "现货收盘价"},
            {"item": "index_1m", "endpoint": "/fapi/v1/indexPriceKlines", "status": "downloaded", "interpretation": "指数价格"},
            {"item": "mark_1m", "endpoint": "/fapi/v1/markPriceKlines", "status": "downloaded", "interpretation": "标记价格"},
        ]
    )
    audit.to_csv(DATA / "第二周接口审计.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    start_ms, end_ms = load_window()
    session = requests.Session()
    session.headers.update({"User-Agent": "funding-settlement-week2/1.0"})
    for kind in ("spot", "index", "mark"):
        frames = []
        for symbol in SYMBOLS:
            symbol_file = DATA / f"{kind}_{symbol}_1m.csv"
            if symbol_file.exists():
                frame = pd.read_csv(symbol_file)
                print(f"Using existing {symbol_file.name}: {len(frame):,} rows")
            else:
                print(f"Downloading {kind} {symbol}")
                frame = klines(session, symbol, start_ms, end_ms, kind)
                frame.to_csv(symbol_file, index=False, encoding="utf-8-sig")
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(DATA / f"{kind}_1m.csv", index=False, encoding="utf-8-sig")
    save_api_audit(session)
    print(f"Saved second-week data to {DATA}")


if __name__ == "__main__":
    main()
