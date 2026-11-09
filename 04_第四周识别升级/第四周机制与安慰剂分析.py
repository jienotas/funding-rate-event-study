# -*- coding: utf-8 -*-
"""Fourth-week local identification upgrade.

This script uses only files already present in [private-dataset]. It aligns the
available BTC 5-minute OI state with event outcomes, reports date-cluster
bootstrap heterogeneity, and creates same-day time-shift placebos. It does
not treat realized funding as a predicted funding variable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WEEK1 = PROJECT / "01_第一周可行性测试"
DATA = ROOT / "data"
OUT = ROOT / "output"
METRICS = PROJECT.parent / "论文_情绪极端性与流动性" / "03_信念持续性估计" / "data" / "binance_metrics_5m_BTCUSDT.csv.gz"
N_BOOT = 4000
SEED = 20261006
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_event_level() -> pd.DataFrame:
    events = pd.read_csv(WEEK1 / "output" / "事件索引.csv")
    events["event_time"] = pd.to_datetime(events["timestamp"], utc=True, format="mixed").dt.floor("min")
    panel = pd.read_csv(WEEK1 / "output" / "事件面板.csv", usecols=["event_id", "tau", "cumulative_bp"])
    wanted = [-60, -10, -1, 0, 60, 120]
    paths = (
        panel.loc[panel["tau"].isin(wanted)]
        .pivot_table(index="event_id", columns="tau", values="cumulative_bp", aggfunc="first")
        .rename(columns={-60: "car_m60", -10: "car_m10", -1: "car_m1", 0: "car_0", 60: "car_p60", 120: "car_p120"})
        .reset_index()
    )
    level = events.merge(paths, on="event_id", how="inner")
    level["event_date"] = level["event_time"].dt.strftime("%Y-%m-%d")
    level["abs_funding_bp"] = pd.to_numeric(level["funding_rate"], errors="coerce").abs() * 10000
    level["pair_id"] = np.where(level["group"].eq("B_pseudo_settlement"), level["event_id"] - 1080, level["event_id"])
    return level


def load_btc_metrics() -> pd.DataFrame:
    if not METRICS.exists():
        return pd.DataFrame()
    cols = [
        "dt_utc", "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
    ]
    oi = pd.read_csv(METRICS, usecols=cols, compression="gzip")
    oi["dt_utc"] = pd.to_datetime(oi["dt_utc"], utc=True, format="mixed").dt.floor("min")
    oi = oi.sort_values("dt_utc").drop_duplicates("dt_utc")
    return oi


def merge_asof_one(events: pd.DataFrame, metrics: pd.DataFrame, offset: str, suffix: str) -> pd.DataFrame:
    if metrics.empty or events.empty:
        return events.copy()
    left = events[["event_id", "event_time"]].copy()
    left["lookup_time"] = left["event_time"] + pd.Timedelta(offset)
    left = left.sort_values("lookup_time")
    right = metrics.rename(columns={"dt_utc": "metric_time"}).sort_values("metric_time")
    merged = pd.merge_asof(left, right, left_on="lookup_time", right_on="metric_time", direction="backward", tolerance=pd.Timedelta(minutes=5))
    value_cols = [c for c in right.columns if c not in {"metric_time"}]
    rename = {c: f"{c}_{suffix}" for c in value_cols}
    merged = merged.rename(columns=rename)
    merged[f"metric_lag_min_{suffix}"] = (merged["lookup_time"] - merged["metric_time"]).dt.total_seconds() / 60
    return merged.drop(columns=["event_time", "lookup_time", "metric_time"], errors="ignore")


def align_oi(level: pd.DataFrame) -> pd.DataFrame:
    oi = load_btc_metrics()
    level = level.copy()
    level["pre_state_time"] = level["event_time"] - pd.Timedelta(minutes=10)
    level["lag_state_time"] = level["event_time"] - pd.Timedelta(minutes=70)
    if oi.empty:
        for col in ["sum_open_interest_value_pre", "sum_open_interest_value_lag", "metric_lag_min_pre", "metric_lag_min_lag"]:
            level[col] = np.nan
        level["oi_source_available"] = False
        return level

    # Only BTC is covered by the existing metrics file; ETH remains explicit missing.
    btc = level.loc[level["symbol"].eq("BTCUSDT")].copy()
    other = level.loc[~level["symbol"].eq("BTCUSDT")].copy()
    pre = merge_asof_one(btc, oi, "-10min", "pre")
    lag = merge_asof_one(btc, oi, "-70min", "lag")
    btc = btc.merge(pre, on="event_id", how="left", suffixes=("", "_prejoin"))
    btc = btc.merge(lag, on="event_id", how="left", suffixes=("", "_lagjoin"))
    for frame in [btc, other]:
        if "sum_open_interest_value_pre" not in frame:
            frame["sum_open_interest_value_pre"] = np.nan
        if "sum_open_interest_value_lag" not in frame:
            frame["sum_open_interest_value_lag"] = np.nan
        for col in ["metric_lag_min_pre", "metric_lag_min_lag"]:
            if col not in frame:
                frame[col] = np.nan
    result = pd.concat([btc, other], ignore_index=True).sort_values("event_id")
    result["pre_oi_value"] = pd.to_numeric(result["sum_open_interest_value_pre"], errors="coerce")
    result["lag_oi_value"] = pd.to_numeric(result["sum_open_interest_value_lag"], errors="coerce")
    result["log_oi_pre"] = np.log(result["pre_oi_value"].where(result["pre_oi_value"] > 0))
    result["dOI_1h_pct"] = 100 * (np.log(result["pre_oi_value"].where(result["pre_oi_value"] > 0)) - np.log(result["lag_oi_value"].where(result["lag_oi_value"] > 0)))
    result["oi_source_available"] = result["pre_oi_value"].notna()
    result.drop(columns=["pre_state_time", "lag_state_time"], inplace=True, errors="ignore")
    return result


def bootstrap_mean(values: pd.DataFrame, value_col: str, cluster_col: str = "event_date") -> dict:
    values = values.dropna(subset=[value_col, cluster_col]).copy()
    if values.empty:
        return {"n": 0, "date_clusters": 0, "mean_bp": np.nan, "lo95_bp": np.nan, "hi95_bp": np.nan, "mde80_bp": np.nan}
    grouped = values.groupby(cluster_col)[value_col].agg(["sum", "count"])
    dates = grouped.index.to_numpy()
    rng = np.random.default_rng(SEED + len(value_col))
    sampled = rng.integers(0, len(dates), size=(N_BOOT, len(dates)))
    draws = grouped["sum"].to_numpy()[sampled].sum(axis=1) / grouped["count"].to_numpy()[sampled].sum(axis=1)
    return {
        "n": int(len(values)), "date_clusters": int(len(dates)), "mean_bp": float(values[value_col].mean()),
        "lo95_bp": float(np.quantile(draws, 0.025)), "hi95_bp": float(np.quantile(draws, 0.975)),
        "mde80_bp": float(2.8 * draws.std(ddof=1)),
    }


def save_oi_heterogeneity(level: pd.DataFrame) -> pd.DataFrame:
    real = level.loc[(level["group"] == "A_real_settlement") & level["symbol"].eq("BTCUSDT")].copy()
    pseudo = level.loc[(level["group"] == "B_pseudo_settlement") & level["symbol"].eq("BTCUSDT")].copy()
    real = real.dropna(subset=["pre_oi_value"]).copy()
    if len(real) < 20 or pseudo.empty:
        return pd.DataFrame()
    real["oi_quartile"] = pd.qcut(real["pre_oi_value"].rank(method="first"), 4, labels=["Q1最低", "Q2", "Q3", "Q4最高"])
    pair = real[["pair_id", "event_date", "oi_quartile", "pre_oi_value", "dOI_1h_pct", "abs_funding_bp", "car_0", "car_p60"]].merge(
        pseudo[["pair_id", "car_0", "car_p60"]].rename(columns={"car_0": "pseudo_car_0", "car_p60": "pseudo_car_p60"}), on="pair_id", how="inner"
    )
    pair["diff_car_0"] = pair["car_0"] - pair["pseudo_car_0"]
    pair["diff_car_p60"] = pair["car_p60"] - pair["pseudo_car_p60"]
    rows = []
    for q, part in pair.groupby("oi_quartile", observed=False):
        for outcome in ["diff_car_0", "diff_car_p60"]:
            row = bootstrap_mean(part, outcome)
            row.update({"oi_quartile": str(q), "outcome": outcome, "pre_oi_median": float(part["pre_oi_value"].median()), "mean_dOI_1h_pct": float(part["dOI_1h_pct"].mean()), "mean_abs_funding_bp": float(part["abs_funding_bp"].mean())})
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第四周BTC_OI异质性_日期聚类bootstrap.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(DATA / "第四周BTC事件级OI对齐.csv", index=False, encoding="utf-8-sig")
    return result


def plot_oi_heterogeneity(result: pd.DataFrame) -> None:
    if result.empty:
        return
    part = result.loc[result["outcome"] == "diff_car_p60"].copy()
    if part.empty:
        return
    order = ["Q1最低", "Q2", "Q3", "Q4最高"]
    part["oi_quartile"] = pd.Categorical(part["oi_quartile"], categories=order, ordered=True)
    part = part.sort_values("oi_quartile")
    x = np.arange(len(part))
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    ax.errorbar(
        x,
        part["mean_bp"],
        yerr=[part["mean_bp"] - part["lo95_bp"], part["hi95_bp"] - part["mean_bp"]],
        fmt="o-",
        color="#B42318",
        capsize=4,
        linewidth=2,
    )
    ax.axhline(0, color="#98A2B3", linewidth=0.8)
    ax.set_xticks(x, part["oi_quartile"].astype(str))
    ax.set_ylabel("真实事件 - 伪事件（基点）")
    ax.set_xlabel("结算前 BTC OI 分位数")
    ax.set_title("第四周：OI 状态与结算后 60 分钟价格差异（探索性）")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "第四周_OI异质性_结算后60分钟.png", bbox_inches="tight")
    plt.close(fig)


def load_prices() -> dict[str, pd.Series]:
    prices = pd.read_csv(WEEK1 / "data" / "perp_1m.csv")
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True, format="mixed").dt.floor("min")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    return {symbol: part.drop_duplicates("timestamp").set_index("timestamp")["price"].sort_index() for symbol, part in prices.groupby("symbol")}


def event_car(series: pd.Series, event_time: pd.Timestamp, taus: list[int]) -> dict | None:
    base_times = event_time + pd.to_timedelta(np.arange(-10, 0), unit="m")
    target_times = event_time + pd.to_timedelta(taus, unit="m")
    base = series.reindex(base_times)
    target = series.reindex(target_times)
    if base.isna().any() or target.isna().any():
        return None
    baseline = float(base.median())
    return {f"car_{t}": float(10000 * np.log(target.loc[event_time + pd.Timedelta(minutes=t)] / baseline)) for t in taus}


def save_placebos(level: pd.DataFrame) -> pd.DataFrame:
    prices = load_prices()
    real = level.loc[level["group"] == "A_real_settlement", ["event_id", "symbol", "event_time", "event_date", "car_0", "car_p60"]].copy()
    rows = []
    for row in real.itertuples(index=False):
        series = prices.get(row.symbol)
        if series is None:
            continue
        for shift in [3, 5]:
            car = event_car(series, row.event_time + pd.Timedelta(hours=shift), [0, 60])
            if car is not None:
                rows.append({"event_id": row.event_id, "symbol": row.symbol, "event_date": row.event_date, "shift_hours": shift, "real_car_0": row.car_0, "real_car_p60": row.car_p60, "placebo_car_0": car["car_0"], "placebo_car_p60": car["car_60"]})
    placebo = pd.DataFrame(rows)
    if placebo.empty:
        return placebo
    placebo["diff_car_0"] = placebo["real_car_0"] - placebo["placebo_car_0"]
    placebo["diff_car_p60"] = placebo["real_car_p60"] - placebo["placebo_car_p60"]
    out = []
    for shift, part in placebo.groupby("shift_hours"):
        for outcome in ["diff_car_0", "diff_car_p60"]:
            row = bootstrap_mean(part, outcome)
            row.update({"shift_hours": shift, "outcome": outcome})
            out.append(row)
    result = pd.DataFrame(out)
    result.to_csv(OUT / "第四周同日时间安慰剂_日期聚类bootstrap.csv", index=False, encoding="utf-8-sig")
    placebo.to_csv(DATA / "第四周同日时间安慰剂事件级.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    level = align_oi(load_event_level())
    level.to_csv(DATA / "第四周事件级机制面板.csv", index=False, encoding="utf-8-sig")
    oi = save_oi_heterogeneity(level)
    plot_oi_heterogeneity(oi)
    placebo = save_placebos(level)
    print("event panel", len(level), "OI matched", int(level["oi_source_available"].sum()))
    print("OI heterogeneity")
    print(oi.to_string(index=False) if not oi.empty else "empty")
    print("placebo")
    print(placebo.to_string(index=False) if not placebo.empty else "empty")


if __name__ == "__main__":
    main()
