# -*- coding: utf-8 -*-
"""Audit which OKX funding-rate field maps to which target settlement."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WEEK2 = ROOT.parent / "02_第二周机制数据管道"
DATA = WEEK2 / "data"
OUT = ROOT / "output"


def main() -> None:
    path = DATA / "okx_funding_rate_archive.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    numeric = [
        "fundingRate", "fundingTime", "nextFundingRate", "nextFundingTime",
        "settFundingRate", "prevFundingTime", "ts",
    ]
    for col in numeric:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["retrieved_at_utc"] = pd.to_datetime(frame["retrieved_at_utc"], utc=True, errors="coerce")
    # Use timezone-aware timestamps directly. Pandas may store parsed datetimes
    # at microsecond rather than nanosecond resolution, so integer conversion
    # is not a safe way to recover milliseconds.
    server_time_utc = pd.to_datetime(frame["ts"], unit="ms", utc=True, errors="coerce")
    frame["funding_time_utc"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True, errors="coerce")
    frame["next_funding_time_utc"] = pd.to_datetime(frame["nextFundingTime"], unit="ms", utc=True, errors="coerce")
    frame["settlement_rate_time_to_retrieval_min"] = (frame["funding_time_utc"] - frame["retrieved_at_utc"]).dt.total_seconds() / 60
    frame["next_rate_time_to_retrieval_min"] = (frame["next_funding_time_utc"] - frame["retrieved_at_utc"]).dt.total_seconds() / 60
    frame["funding_rate_nonmissing"] = frame["fundingRate"].notna()
    frame["next_funding_rate_nonmissing"] = frame["nextFundingRate"].notna()
    frame["settlement_rate_nonmissing"] = frame["settFundingRate"].notna()
    frame["rate_field_used_for_current_target"] = np.where(frame["fundingRate"].notna(), "fundingRate", "unavailable")
    frame["target_settlement_time_for_used_field"] = frame["funding_time_utc"].astype("string")
    frame["collection_delay_sec"] = (server_time_utc - frame["retrieved_at_utc"]).dt.total_seconds()
    columns = [
        "instId", "retrieved_at_utc", "ts", "collection_delay_sec", "funding_time_utc",
        "next_funding_time_utc", "settlement_rate_time_to_retrieval_min", "next_rate_time_to_retrieval_min",
        "fundingRate", "nextFundingRate", "settFundingRate", "funding_rate_nonmissing",
        "next_funding_rate_nonmissing", "settlement_rate_nonmissing", "rate_field_used_for_current_target",
        "target_settlement_time_for_used_field", "settState", "method",
    ]
    audit = frame[[c for c in columns if c in frame.columns]].sort_values(["instId", "retrieved_at_utc"])
    audit.to_csv(OUT / "第四周OKX字段语义审计.csv", index=False, encoding="utf-8-sig")
    dedup = frame.drop_duplicates(["instId", "ts"]).copy()
    summary = pd.DataFrame(
        [
            {"metric": "raw_rows", "value": len(frame)},
            {"metric": "deduplicated_rows_by_instId_ts", "value": len(dedup)},
            {"metric": "instruments", "value": frame["instId"].nunique()},
            {"metric": "fundingRate_nonmissing_share", "value": frame["fundingRate"].notna().mean()},
            {"metric": "nextFundingRate_nonmissing_share", "value": frame["nextFundingRate"].notna().mean()},
            {"metric": "settFundingRate_nonmissing_share", "value": frame["settFundingRate"].notna().mean()},
            {"metric": "fundingRate_target_median_minutes_ahead", "value": frame["settlement_rate_time_to_retrieval_min"].median()},
            {"metric": "nextFundingRate_target_median_minutes_ahead", "value": frame["next_rate_time_to_retrieval_min"].median()},
        ]
    )
    summary.to_csv(OUT / "第四周OKX字段语义审计摘要.csv", index=False, encoding="utf-8-sig")
    print(audit.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
