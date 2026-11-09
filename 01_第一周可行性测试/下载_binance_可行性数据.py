# -*- coding: utf-8 -*-
"""Download a bounded Binance USDT-margined perpetual sample for the event plots.

This downloader intentionally collects realized funding history. Binance's public
history endpoint does not by itself establish that the rate was observable before
settlement, so the resulting figures are exploratory only.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://fapi.binance.com"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
MINUTE_MS = 60_000
KLINE_LIMIT = 1_500


def get_json(session: requests.Session, path: str, params: dict) -> list:
    response = session.get(BASE_URL + path, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("code", 0) < 0:
        raise RuntimeError(f"Binance API error: {payload}")
    return payload


def funding_history(session: requests.Session, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        payload = get_json(
            session,
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1_000},
        )
        if not payload:
            break
        rows.extend(payload)
        last = int(payload[-1]["fundingTime"])
        if last < cursor:
            break
        cursor = last + 1
        if len(payload) < 1_000:
            break
        time.sleep(0.05)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No funding rows returned for {symbol}")
    frame = frame.rename(columns={"fundingTime": "timestamp", "fundingRate": "funding_rate"})
    frame["symbol"] = symbol
    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True)
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="coerce")
    frame = frame[["timestamp", "symbol", "funding_rate"]].drop_duplicates()
    return frame.sort_values("timestamp")


def minute_klines(session: requests.Session, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    cursor = start_ms
    request_count = 0
    while cursor < end_ms:
        payload = get_json(
            session,
            "/fapi/v1/continuousKlines",
            {
                "pair": symbol,
                "contractType": "PERPETUAL",
                "interval": "1m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": KLINE_LIMIT,
            },
        )
        request_count += 1
        if not payload:
            break
        rows.extend(payload)
        last = int(payload[-1][0])
        if last < cursor:
            break
        cursor = last + MINUTE_MS
        if request_count % 20 == 0:
            print(f"{symbol}: downloaded {len(rows):,} minute rows")
        time.sleep(0.08)
    if not rows:
        raise RuntimeError(f"No kline rows returned for {symbol}")
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={0: "timestamp", 4: "price"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True)
    frame["symbol"] = symbol
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame[["timestamp", "symbol", "price"]].drop_duplicates(["timestamp", "symbol"])
    return frame.sort_values("timestamp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--end", type=str, default="", help="UTC ISO time; default is now")
    args = parser.parse_args()
    if args.days <= 0:
        raise ValueError("--days must be positive")

    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=args.days)
    start = start.floor("min")
    end = end.ceil("min")
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "funding-settlement-feasibility/1.0"})

    funding_frames = []
    price_frames = []
    for symbol in SYMBOLS:
        print(f"Downloading {symbol} funding history: {start.isoformat()} to {end.isoformat()}")
        funding_frames.append(funding_history(session, symbol, start_ms, end_ms))
        print(f"Downloading {symbol} 1m perpetual klines")
        price_frames.append(minute_klines(session, symbol, start_ms, end_ms))

    funding = pd.concat(funding_frames, ignore_index=True)
    prices = pd.concat(price_frames, ignore_index=True)
    funding.to_csv(data_dir / "funding.csv", index=False, encoding="utf-8-sig")
    prices.to_csv(data_dir / "perp_1m.csv", index=False, encoding="utf-8-sig")
    metadata = pd.DataFrame(
        [
            {"field": "source", "value": "Binance Futures public REST API"},
            {"field": "funding_endpoint", "value": "/fapi/v1/fundingRate (realized funding history)"},
            {"field": "price_endpoint", "value": "/fapi/v1/continuousKlines (PERPETUAL, 1m close)"},
            {"field": "start_utc", "value": start.isoformat()},
            {"field": "end_utc", "value": end.isoformat()},
            {"field": "symbols", "value": ",".join(SYMBOLS)},
            {"field": "warning", "value": "Realized funding is not proof of pre-settlement predicted funding; figures are exploratory."},
        ]
    )
    metadata.to_csv(data_dir / "下载元数据.csv", index=False, encoding="utf-8-sig")
    print(f"Saved funding rows: {len(funding):,}")
    print(f"Saved price rows: {len(prices):,}")
    print(f"Output directory: {data_dir}")


if __name__ == "__main__":
    main()
