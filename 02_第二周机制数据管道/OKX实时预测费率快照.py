# -*- coding: utf-8 -*-
"""Capture OKX's public funding-rate fields for future pre-settlement archiving."""

from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
URL = "https://www.okx.com/api/v5/public/funding-rate"
INSTRUMENTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]


def main() -> None:
    rows = []
    session = requests.Session()
    session.headers.update({"User-Agent": "funding-settlement-week2/1.0"})
    for inst_id in INSTRUMENTS:
        response = session.get(URL, params={"instId": inst_id}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "0" or not payload.get("data"):
            raise RuntimeError(f"OKX API response: {payload}")
        row = dict(payload["data"][0])
        row["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        rows.append(row)
    frame = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA / "okx_funding_rate_snapshot.csv", index=False, encoding="utf-8-sig")
    print(frame[["instId", "fundingRate", "nextFundingRate", "fundingTime", "nextFundingTime", "settFundingRate", "settState", "premium"]].to_string(index=False))


if __name__ == "__main__":
    main()
