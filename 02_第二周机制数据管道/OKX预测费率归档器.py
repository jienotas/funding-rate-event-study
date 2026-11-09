# -*- coding: utf-8 -*-
"""Archive OKX pre-settlement fields at a fixed interval.

Run without arguments for one snapshot. Use --minutes to run a bounded local
archive; the script never sends orders or requires an account.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
URL = "https://www.okx.com/api/v5/public/funding-rate"
INSTRUMENTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]


def fetch(session: requests.Session) -> pd.DataFrame:
    rows = []
    retrieved = pd.Timestamp.now(tz="UTC")
    for inst_id in INSTRUMENTS:
        response = session.get(URL, params={"instId": inst_id}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "0" or not payload.get("data"):
            raise RuntimeError(payload)
        row = dict(payload["data"][0])
        row["retrieved_at_utc"] = retrieved.isoformat()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=0, help="bounded archive duration; 0 means one snapshot")
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.minutes < 0 or args.interval_seconds < 10:
        raise ValueError("minutes must be nonnegative and interval-seconds must be >= 10")

    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "okx_funding_rate_archive.csv"
    session = requests.Session()
    session.headers.update({"User-Agent": "funding-settlement-week2/1.0"})
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
    first = True
    while first or (deadline is not None and time.monotonic() < deadline):
        first = False
        frame = fetch(session)
        frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
        print(frame[["instId", "fundingRate", "nextFundingRate", "nextFundingTime", "settFundingRate", "settState"]].to_string(index=False))
        if deadline is None or time.monotonic() + args.interval_seconds >= deadline:
            break
        time.sleep(args.interval_seconds)
    print(f"Saved archive: {path}")


if __name__ == "__main__":
    main()
