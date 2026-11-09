# -*- coding: utf-8 -*-
"""Download Binance 5-minute perpetual liquidity proxies and test mechanisms."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WEEK1 = PROJECT / "01_第一周可行性测试"
WEEK4 = PROJECT / "04_第四周识别升级"
WEEK5 = PROJECT / "05_第五周正式机制模型"
DATA = ROOT / "data"
OUT = ROOT / "output"
URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "5m"
LIMIT = 1000
N_BOOT = 4000
SEED = 20261006


def event_times() -> tuple[pd.Timestamp, pd.Timestamp]:
    events = pd.read_csv(WEEK1 / "output" / "事件索引.csv")
    t = pd.to_datetime(events["timestamp"], utc=True, format="mixed").dt.floor("min")
    return t.min() - pd.Timedelta(hours=2), t.max() + pd.Timedelta(hours=3)


def download_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[list] = []
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    session = requests.Session()
    session.headers.update({"User-Agent": "funding-settlement-week6/1.0"})
    while start_ms < end_ms:
        response = session.get(
            URL,
            params={"symbol": symbol, "interval": INTERVAL, "startTime": start_ms, "endTime": end_ms, "limit": LIMIT},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            break
        rows.extend(payload)
        last = int(payload[-1][0])
        if last < start_ms:
            break
        start_ms = last + 5 * 60 * 1000
        if len(payload) < LIMIT:
            break
        time.sleep(0.08)
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame["symbol"] = symbol
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["trades"] = pd.to_numeric(frame["trades"], errors="coerce")
    return frame.drop_duplicates(["symbol", "timestamp"]).sort_values("timestamp")


def download_all() -> pd.DataFrame:
    path = DATA / "binance_futures_5m_liquidity.csv"
    if path.exists():
        existing = pd.read_csv(path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, format="mixed")
        if set(SYMBOLS).issubset(set(existing["symbol"].unique())):
            return existing
    start, end = event_times()
    frames = [download_symbol(symbol, start, end) for symbol in SYMBOLS]
    frame = pd.concat(frames, ignore_index=True)
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"symbol": symbol, "start_utc": start.isoformat(), "end_utc": end.isoformat(), "rows": int((frame["symbol"] == symbol).sum())} for symbol in SYMBOLS]
    ).to_csv(DATA / "第六周流动性下载元数据.csv", index=False, encoding="utf-8-sig")
    return frame


def load_events() -> pd.DataFrame:
    events = pd.read_csv(WEEK1 / "output" / "事件索引.csv")
    events["event_time"] = pd.to_datetime(events["timestamp"], utc=True, format="mixed").dt.floor("min")
    events["event_date"] = events["event_time"].dt.strftime("%Y-%m-%d")
    events["event_hour_utc"] = events["event_time"].dt.hour.astype(str)
    events["pair_id"] = np.where(events["group"].eq("B_pseudo_settlement"), events["event_id"] - 1080, events["event_id"])
    events["abs_funding_bp"] = pd.to_numeric(events["funding_rate"], errors="coerce").abs() * 10000
    panel = pd.read_csv(WEEK1 / "output" / "事件面板.csv", usecols=["event_id", "tau", "cumulative_bp"])
    paths = panel.loc[panel["tau"].isin([0, 60])].pivot_table(index="event_id", columns="tau", values="cumulative_bp", aggfunc="first").rename(columns={0: "car_0", 60: "car_p60"}).reset_index()
    return events.merge(paths, on="event_id", how="inner")


def build_event_liquidity(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, format="mixed")
    by_key = {(symbol): part.set_index("timestamp").sort_index() for symbol, part in bars.groupby("symbol")}
    rows = []
    for event in events.itertuples(index=False):
        part = by_key.get(event.symbol)
        if part is None:
            continue
        start = event.event_time - pd.Timedelta(minutes=60)
        stop = event.event_time - pd.Timedelta(minutes=5)
        window = part.loc[(part.index >= start) & (part.index <= stop)]
        post = part.loc[(part.index >= event.event_time) & (part.index < event.event_time + pd.Timedelta(minutes=5))]
        if len(window) < 8:
            continue
        quote = pd.to_numeric(window["quote_volume"], errors="coerce").fillna(0)
        taker = pd.to_numeric(window["taker_buy_quote"], errors="coerce").fillna(0)
        high = pd.to_numeric(window["high"], errors="coerce")
        low = pd.to_numeric(window["low"], errors="coerce")
        trades = pd.to_numeric(window["trades"], errors="coerce").fillna(0)
        qsum = quote.sum()
        post_quote = pd.to_numeric(post["quote_volume"], errors="coerce").fillna(0)
        post_taker = pd.to_numeric(post["taker_buy_quote"], errors="coerce").fillna(0)
        post_high = pd.to_numeric(post["high"], errors="coerce")
        post_low = pd.to_numeric(post["low"], errors="coerce")
        post_qsum = post_quote.sum()
        post_imbalance = (2 * post_taker.sum() - post_qsum) / post_qsum if post_qsum > 0 else np.nan
        rows.append(
            {
                "event_id": event.event_id, "pair_id": event.pair_id, "symbol": event.symbol, "group": event.group, "event_date": event.event_date, "event_hour_utc": event.event_hour_utc,
                "event_time": event.event_time, "car_0": event.car_0, "car_p60": event.car_p60, "abs_funding_bp": event.abs_funding_bp,
                "pre_quote_volume": qsum, "pre_quote_volume_log": np.log1p(qsum),
                "pre_taker_imbalance": (2 * taker.sum() - qsum) / qsum if qsum > 0 else np.nan,
                "pre_range_bp": float((10000 * np.log(high / low)).mean()),
                "pre_trades_log": np.log1p(trades.sum()), "pre_bar_count": len(window),
                "post_quote_volume_log": np.log1p(post_qsum) if post_qsum > 0 else np.nan,
                "post_taker_imbalance": post_imbalance,
                "post_range_bp": float((10000 * np.log(post_high / post_low)).mean()) if len(post) else np.nan,
                "post_bar_count": len(post),
            }
        )
    result = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    result.to_csv(DATA / "第六周事件级流动性面板.csv", index=False, encoding="utf-8-sig")
    return result


def bootstrap_mean(frame: pd.DataFrame, outcome: str) -> dict:
    frame = frame.dropna(subset=[outcome, "event_date"])
    grouped = frame.groupby("event_date")[outcome].agg(["sum", "count"])
    dates = grouped.index.to_numpy()
    rng = np.random.default_rng(SEED + sum(map(ord, outcome)))
    sampled = rng.integers(0, len(dates), size=(N_BOOT, len(dates)))
    draws = grouped["sum"].to_numpy()[sampled].sum(axis=1) / grouped["count"].to_numpy()[sampled].sum(axis=1)
    return {"n": len(frame), "date_clusters": len(dates), "mean": float(frame[outcome].mean()), "lo95": float(np.quantile(draws, 0.025)), "hi95": float(np.quantile(draws, 0.975)), "mde80": float(2.8 * draws.std(ddof=1))}


def paired_mechanism(liquidity: pd.DataFrame) -> pd.DataFrame:
    real = liquidity.loc[liquidity["group"].eq("A_real_settlement")].copy()
    pseudo = liquidity.loc[liquidity["group"].eq("B_pseudo_settlement")].copy()
    rcols = ["pair_id", "symbol", "event_hour_utc", "event_date", "car_0", "car_p60", "abs_funding_bp", "pre_quote_volume_log", "pre_taker_imbalance", "pre_range_bp", "pre_trades_log", "post_quote_volume_log", "post_taker_imbalance", "post_range_bp"]
    pcols = ["pair_id", "car_0", "car_p60", "pre_quote_volume_log", "pre_taker_imbalance", "pre_range_bp", "pre_trades_log", "post_quote_volume_log", "post_taker_imbalance", "post_range_bp"]
    pair = real[rcols].merge(pseudo[pcols].rename(columns={c: f"pseudo_{c}" for c in pcols if c != "pair_id"}), on="pair_id", how="inner")
    pair["diff_car_0"] = pair["car_0"] - pair["pseudo_car_0"]
    pair["diff_car_p60"] = pair["car_p60"] - pair["pseudo_car_p60"]
    for col in ["post_quote_volume_log", "post_taker_imbalance", "post_range_bp"]:
        pair[f"diff_{col}"] = pair[col] - pair[f"pseudo_{col}"]
    pair.to_csv(DATA / "第六周配对流动性机制面板.csv", index=False, encoding="utf-8-sig")
    rows = []
    for outcome in ["diff_car_0", "diff_car_p60"]:
        summary = bootstrap_mean(pair, outcome)
        summary.update({"outcome": outcome, "model": "paired mean"})
        rows.append(summary)
        model_data = pair.dropna(subset=[outcome, "pre_quote_volume_log", "pre_taker_imbalance", "pre_range_bp", "pre_trades_log", "abs_funding_bp"]).copy()
        for col in ["pre_quote_volume_log", "pre_taker_imbalance", "pre_range_bp", "pre_trades_log"]:
            std = model_data[col].std(ddof=1)
            model_data[f"z_{col}"] = (model_data[col] - model_data[col].mean()) / std if std else 0
        model = smf.ols(f"{outcome} ~ z_pre_quote_volume_log + z_pre_taker_imbalance + z_pre_range_bp + z_pre_trades_log + abs_funding_bp + C(symbol) + C(event_hour_utc)", data=model_data).fit(cov_type="cluster", cov_kwds={"groups": model_data["event_date"]})
        for term in ["z_pre_quote_volume_log", "z_pre_taker_imbalance", "z_pre_range_bp", "z_pre_trades_log", "abs_funding_bp"]:
            rows.append({"outcome": outcome, "model": "clustered regression", "term": term, "coefficient_bp": model.params.get(term), "cluster_se_bp": model.bse.get(term), "p_value": model.pvalues.get(term), "ci95_lo_bp": model.conf_int().loc[term, 0], "ci95_hi_bp": model.conf_int().loc[term, 1], "n": int(model.nobs), "date_clusters": int(model_data["event_date"].nunique()), "r_squared": model.rsquared})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第六周流动性机制结果.csv", index=False, encoding="utf-8-sig")
    post_rows = []
    for outcome in ["diff_post_quote_volume_log", "diff_post_taker_imbalance", "diff_post_range_bp"]:
        summary = bootstrap_mean(pair, outcome)
        summary.update({"outcome": outcome, "model": "paired post-state difference"})
        post_rows.append(summary)
    post_result = pd.DataFrame(post_rows)
    post_result.to_csv(OUT / "第六周结算后流动性变化.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    bars = download_all()
    if bars.empty:
        raise RuntimeError("Binance liquidity download returned no rows; no mechanism estimates were produced")
    events = load_events()
    liquidity = build_event_liquidity(events, bars)
    if liquidity.empty:
        raise RuntimeError("No event liquidity windows were available")
    pd.DataFrame(
        [{"metric": "bars", "value": len(bars)}, {"metric": "event_liquidity_rows", "value": len(liquidity)}, {"metric": "pre_window_complete_share", "value": float((liquidity["pre_bar_count"] >= 12).mean())}, {"metric": "post_5m_complete_share", "value": float((liquidity["post_bar_count"] == 1).mean())}]
    ).to_csv(OUT / "第六周流动性数据质量摘要.csv", index=False, encoding="utf-8-sig")
    result = paired_mechanism(liquidity)
    print("bars", len(bars), "event liquidity rows", len(liquidity))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
