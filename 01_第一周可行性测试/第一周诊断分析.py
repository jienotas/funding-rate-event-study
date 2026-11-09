# -*- coding: utf-8 -*-
"""Complete the exploratory first-week diagnostics after the three event plots.

The output is deliberately descriptive: it uses realized funding, simple event
means, and transparent confidence intervals. It is not the final causal model.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
METRICS = Path(os.environ.get("METRICS_CSV", ROOT / "data" / "binance_metrics_5m_BTCUSDT.csv.gz"))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def mean_ci(values: pd.Series) -> tuple[float, float, float, int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = int(values.size)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return mean, mean - 1.96 * se, mean + 1.96 * se, n


def load_event_level() -> pd.DataFrame:
    events = pd.read_csv(OUT / "事件索引.csv")
    panel = pd.read_csv(OUT / "事件面板.csv")
    paths = (
        panel.loc[panel["tau"].isin([-60, -30, -10, -1, 0, 30, 60, 120])]
        .pivot_table(index="event_id", columns="tau", values="cumulative_bp", aggfunc="first")
        .rename(
            columns={
                -60: "car_m60",
                -30: "car_m30",
                -10: "car_m10",
                -1: "car_m1",
                0: "car_0",
                30: "car_p30",
                60: "car_p60",
                120: "car_p120",
            }
        )
        .reset_index()
    )
    level = events.merge(paths, on="event_id", how="inner")
    level["event_time"] = pd.to_datetime(level["timestamp"], utc=True, format="mixed")
    level["event_date"] = level["event_time"].dt.date.astype(str)
    level["abs_funding_bp"] = level["funding_rate"].abs() * 10000
    return level


def save_summary(level: pd.DataFrame) -> pd.DataFrame:
    outcomes = ["car_m60", "car_m30", "car_m1", "car_0", "car_p30", "car_p60", "car_p120"]
    rows = []
    for group, part in level.groupby("group"):
        for outcome in outcomes:
            mean, lo, hi, n = mean_ci(part[outcome])
            rows.append({"group": group, "outcome": outcome, "mean_bp": mean, "lo95": lo, "hi95": hi, "count": n})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "第一周事件响应汇总.csv", index=False, encoding="utf-8-sig")

    diffs = []
    a = level.loc[level["group"] == "A_real_settlement"]
    b = level.loc[level["group"] == "B_pseudo_settlement"]
    for outcome in outcomes:
        av = pd.to_numeric(a[outcome], errors="coerce").dropna()
        bv = pd.to_numeric(b[outcome], errors="coerce").dropna()
        diff = av.mean() - bv.mean()
        se = np.sqrt(av.var(ddof=1) / len(av) + bv.var(ddof=1) / len(bv))
        diffs.append(
            {
                "outcome": outcome,
                "real_minus_pseudo_bp": diff,
                "lo95": diff - 1.96 * se,
                "hi95": diff + 1.96 * se,
                "real_count": len(av),
                "pseudo_count": len(bv),
            }
        )
    diff_frame = pd.DataFrame(diffs)
    diff_frame.to_csv(OUT / "真实减伪事件差异.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_grouped_response(level: pd.DataFrame, column: str, category: str, filename: str, title: str) -> None:
    rows = []
    for group, part in level.groupby(category, observed=False):
        for horizon, outcome in [("结算时点", "car_0"), ("结算后 60 分钟", "car_p60")]:
            mean, lo, hi, n = mean_ci(part[outcome])
            rows.append({"category": str(group), "horizon": horizon, "mean": mean, "lo": lo, "hi": hi, "count": n})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"{filename}.csv", index=False, encoding="utf-8-sig")
    categories = list(result["category"].drop_duplicates())
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    colors = {"结算时点": "#B42318", "结算后 60 分钟": "#1570EF"}
    for horizon in colors:
        part = result.loc[result["horizon"] == horizon].set_index("category").reindex(categories)
        ax.errorbar(
            x,
            part["mean"],
            yerr=[part["mean"] - part["lo"], part["hi"] - part["mean"]],
            marker="o",
            linewidth=2,
            capsize=4,
            color=colors[horizon],
            label=horizon,
        )
    ax.axhline(0, color="#98A2B3", linewidth=0.8)
    ax.set_xticks(x, categories)
    ax.set_xlabel(column)
    ax.set_ylabel("累计价格变化（基点）")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / filename.replace(".csv", ".png"), bbox_inches="tight")
    plt.close(fig)


def add_sign_and_oi(level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    real = level.loc[level["group"] == "A_real_settlement"].copy()
    real["费率方向"] = np.where(real["funding_rate"] >= 0, "正费率", "负费率")
    plot_grouped_response(real, "资金费率方向", "费率方向", "第一周_正负费率响应", "正负资金费率的事件响应")

    if not METRICS.exists():
        return real, pd.DataFrame()
    oi = pd.read_csv(METRICS, usecols=["dt_utc", "sum_open_interest_value"])
    oi["dt_utc"] = pd.to_datetime(oi["dt_utc"], utc=True, format="mixed")
    oi = oi.drop_duplicates("dt_utc").sort_values("dt_utc")
    btc = real.loc[real["symbol"] == "BTCUSDT"].copy()
    btc["pre_state_time"] = btc["event_time"] - pd.Timedelta(minutes=10)
    btc = pd.merge_asof(
        btc.sort_values("pre_state_time"),
        oi,
        left_on="pre_state_time",
        right_on="dt_utc",
        direction="backward",
        tolerance=pd.Timedelta(minutes=15),
    )
    btc = btc.rename(columns={"sum_open_interest_value": "pre_oi_value"})
    btc = btc.dropna(subset=["pre_oi_value"])
    if len(btc) >= 16:
        btc["OI状态"] = pd.qcut(
            btc["pre_oi_value"].rank(method="first"),
            q=4,
            labels=["OI最低", "OI中低", "OI中高", "OI最高"],
        )
        plot_grouped_response(btc, "结算前 OI 分位数", "OI状态", "第一周_OI状态响应", "结算前 OI 状态与事件响应（BTCUSDT）")
    return real, btc


def save_quality(level: pd.DataFrame) -> None:
    funding = pd.read_csv(DATA / "funding.csv")
    prices = pd.read_csv(DATA / "perp_1m.csv")
    funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True, format="mixed")
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True, format="mixed")
    intervals = funding.sort_values(["symbol", "timestamp"]).groupby("symbol")["timestamp"].diff().dt.total_seconds() / 60
    interval_by_symbol = funding.assign(interval_minutes=intervals).groupby("symbol")["interval_minutes"].agg(["count", "median", "min", "max"])
    rows = [
        {"metric": "funding_rows", "value": len(funding)},
        {"metric": "price_rows", "value": len(prices)},
        {"metric": "funding_start_utc", "value": funding["timestamp"].min().isoformat()},
        {"metric": "funding_end_utc", "value": funding["timestamp"].max().isoformat()},
        {"metric": "price_start_utc", "value": prices["timestamp"].min().isoformat()},
        {"metric": "price_end_utc", "value": prices["timestamp"].max().isoformat()},
        {"metric": "valid_real_events", "value": int((level["group"] == "A_real_settlement").sum())},
        {"metric": "valid_pseudo_events", "value": int((level["group"] == "B_pseudo_settlement").sum())},
        {"metric": "valid_near_zero_events", "value": int((level["group"] == "C_near_zero_funding").sum())},
    ]
    for symbol, row in interval_by_symbol.iterrows():
        rows.extend(
            [
                {"metric": f"{symbol}_interval_median_minutes", "value": row["median"]},
                {"metric": f"{symbol}_interval_min_minutes", "value": row["min"]},
                {"metric": f"{symbol}_interval_max_minutes", "value": row["max"]},
            ]
        )
    pd.DataFrame(rows).to_csv(OUT / "第一周数据质量与规则检查.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    level = load_event_level()
    save_summary(level)
    real, btc_oi = add_sign_and_oi(level)
    save_quality(level)
    print("event_level_rows", len(level))
    print("real_rows", len(real), "btc_oi_rows", len(btc_oi))
    print(level.groupby("group")["car_p60"].agg(["mean", "count"]).to_string())


if __name__ == "__main__":
    main()
