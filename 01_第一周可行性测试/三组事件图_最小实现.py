# -*- coding: utf-8 -*-
"""资金费率结算的三组事件路径最小可行性测试。

输入：data/funding.csv, data/perp_1m.csv
输出：output/ 下的两张图、事件面板和样本量表。

该脚本只做事件对齐和可视化，不下载交易所数据，也不把图解释为因果结果。
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
FUNDING_FILE = DATA_DIR / "funding.csv"
PRICE_FILE = DATA_DIR / "perp_1m.csv"

WINDOW_MINUTES = 120
ZERO_RATE_THRESHOLD = 1e-5  # 0.001%; funding_rate 使用小数表示
MAX_FORWARD_FILL_MINUTES = 1

# Make the generated figures readable on the Windows environment used for this
# feasibility test. Fall back to DejaVu Sans if the Chinese fonts are absent.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_utc(values):
    """同时支持 ISO 字符串和交易所常见的 Unix 毫秒时间戳。"""
    raw = pd.Series(values)
    numeric = pd.to_numeric(raw, errors="coerce")
    # API exports mix timestamps with and without fractional seconds. The
    # mixed parser prevents valid rows from being dropped because pandas inferred
    # one format from the first row only.
    try:
        parsed = pd.to_datetime(raw, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    mask = numeric.notna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(numeric.loc[mask], unit="ms", utc=True)
    return parsed


def read_input():
    if not FUNDING_FILE.exists() or not PRICE_FILE.exists():
        raise FileNotFoundError(
            "请先准备 data/funding.csv 和 data/perp_1m.csv。字段说明见数据字段说明.md"
        )

    funding = pd.read_csv(FUNDING_FILE)
    prices = pd.read_csv(PRICE_FILE)
    required_funding = {"timestamp", "symbol", "funding_rate"}
    required_prices = {"timestamp", "symbol", "price"}
    if not required_funding.issubset(funding.columns):
        raise ValueError(f"funding.csv 缺少字段：{required_funding - set(funding.columns)}")
    if not required_prices.issubset(prices.columns):
        raise ValueError(f"perp_1m.csv 缺少字段：{required_prices - set(prices.columns)}")

    funding["timestamp"] = parse_utc(funding["timestamp"])
    prices["timestamp"] = parse_utc(prices["timestamp"])
    # Binance funding timestamps may carry a few milliseconds after the hour.
    # The 1-minute kline index is minute-aligned, so normalize event timestamps
    # before constructing the event window.
    funding["timestamp"] = funding["timestamp"].dt.floor("min")
    prices["timestamp"] = prices["timestamp"].dt.floor("min")
    funding["funding_rate"] = pd.to_numeric(funding["funding_rate"], errors="coerce")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    funding = funding.dropna(subset=["timestamp", "symbol", "funding_rate"])
    prices = prices.dropna(subset=["timestamp", "symbol", "price"])
    funding = funding.drop_duplicates(["symbol", "timestamp"])
    prices = prices.drop_duplicates(["symbol", "timestamp"])
    funding = funding.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return funding, prices


def infer_intervals(funding):
    """按合约估计结算周期，并用中位数填补每个事件的周期。"""
    funding = funding.copy()
    funding["interval_minutes"] = (
        funding.groupby("symbol")["timestamp"].diff().dt.total_seconds() / 60
    )
    medians = funding.groupby("symbol")["interval_minutes"].transform("median")
    funding["interval_minutes"] = funding["interval_minutes"].fillna(medians)
    return funding


def make_events(funding):
    """生成真实事件、伪事件和近零费率事件。"""
    funding = infer_intervals(funding)
    real = funding[["symbol", "timestamp", "funding_rate", "interval_minutes"]].copy()
    real["group"] = "A_real_settlement"

    pseudo = real.copy()
    pseudo["timestamp"] = pseudo["timestamp"] + pd.to_timedelta(
        pseudo["interval_minutes"] / 2, unit="m"
    )
    pseudo["funding_rate"] = np.nan
    pseudo["group"] = "B_pseudo_settlement"

    near_zero = real.loc[real["funding_rate"].abs() <= ZERO_RATE_THRESHOLD].copy()
    near_zero["group"] = "C_near_zero_funding"

    events = pd.concat([real, pseudo, near_zero], ignore_index=True)
    events["event_id"] = np.arange(len(events))
    return events


def price_series(prices, symbol):
    """将价格整理为 1 分钟索引，并只允许极短缺口前向填充。"""
    p = prices.loc[prices["symbol"] == symbol, ["timestamp", "price"]].copy()
    if p.empty:
        return None
    p = p.set_index("timestamp")["price"].sort_index()
    # 如果原始时间戳是分钟开盘或分钟收盘，统一到分钟频率。
    p = p.resample("1min").last()
    return p.ffill(limit=MAX_FORWARD_FILL_MINUTES)


def extract_event_panel(events, prices):
    tau_values = np.arange(-WINDOW_MINUTES, WINDOW_MINUTES + 1)
    price_cache = {symbol: price_series(prices, symbol) for symbol in events["symbol"].unique()}
    rows = []
    skipped = []
    for event in events.itertuples(index=False):
        series = price_cache.get(event.symbol)
        if series is None:
            skipped.append((event.event_id, "missing_symbol"))
            continue
        targets = event.timestamp + pd.to_timedelta(tau_values, unit="m")
        window = series.reindex(targets)
        if window.isna().any() or (window <= 0).any():
            skipped.append((event.event_id, "incomplete_window"))
            continue

        # 用事件前 [-10,-1] 分钟中间值作基准，避免把 tau=-120 误写成
        # 事件前基准，也降低单分钟价格噪声对路径的影响。
        pre_event = window.loc[window.index <= event.timestamp - pd.Timedelta(minutes=1)]
        pre_event = pre_event.tail(10)
        base = float(pre_event.median()) if not pre_event.empty else float(window.iloc[0])
        cumulative_bp = 10000 * np.log(window.to_numpy(dtype=float) / base)
        minute_bp = np.r_[np.nan, np.diff(cumulative_bp)]
        for tau, car, one_minute in zip(tau_values, cumulative_bp, minute_bp):
            rows.append(
                {
                    "event_id": event.event_id,
                    "symbol": event.symbol,
                    "event_time": event.timestamp,
                    "group": event.group,
                    "funding_rate": event.funding_rate,
                    "tau": int(tau),
                    "cumulative_bp": float(car),
                    "minute_return_bp": float(one_minute)
                    if np.isfinite(one_minute)
                    else np.nan,
                }
            )
    if skipped:
        warnings.warn(f"跳过 {len(skipped)} 个事件；原因示例：{skipped[:3]}")
    return pd.DataFrame(rows)


def summarize(panel):
    """事件层面的均值和近似 95% 置信区间。"""
    summary = (
        panel.groupby(["group", "tau"])[["cumulative_bp", "minute_return_bp"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    for metric in ["cumulative_bp", "minute_return_bp"]:
        mean = summary[(metric, "mean")]
        std = summary[(metric, "std")]
        count = summary[(metric, "count")]
        summary[(metric, "lo95")] = mean - 1.96 * std / np.sqrt(count)
        summary[(metric, "hi95")] = mean + 1.96 * std / np.sqrt(count)
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    return summary


def plot_metric(summary, metric, ylabel, filename):
    colors = {
        "A_real_settlement": "#B42318",
        "B_pseudo_settlement": "#667085",
        "C_near_zero_funding": "#1570EF",
    }
    labels = {
        "A_real_settlement": "A 真实结算",
        "B_pseudo_settlement": "B 伪结算",
        "C_near_zero_funding": "C 近零费率结算",
    }
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    for group in colors:
        part = summary.loc[summary["group"] == group].sort_values("tau")
        if part.empty:
            continue
        x = part["tau"].to_numpy()
        y = part[f"{metric}_mean"].to_numpy()
        lo = part[f"{metric}_lo95"].to_numpy()
        hi = part[f"{metric}_hi95"].to_numpy()
        ax.plot(x, y, color=colors[group], linewidth=2, label=labels[group])
        ax.fill_between(x, lo, hi, color=colors[group], alpha=0.12)
    ax.axvline(0, color="#111827", linestyle="--", linewidth=1)
    ax.axhline(0, color="#98A2B3", linewidth=0.8)
    ax.set_xlabel("相对事件时间 tau（分钟）")
    ax.set_ylabel(ylabel)
    ax.set_title("资金费率结算三组事件路径")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_dose_response(panel):
    """Plot event-level responses by absolute realized funding-rate quartile.

    This is a feasibility diagnostic, not a causal estimate. It checks whether
    the event path changes monotonically with funding-rate magnitude.
    """
    real = panel.loc[panel["group"] == "A_real_settlement"].copy()
    path = (
        real.loc[real["tau"].isin([0, 60])]
        .pivot_table(index="event_id", columns="tau", values="cumulative_bp", aggfunc="first")
        .rename(columns={0: "tau_0", 60: "tau_60"})
    )
    meta = real.groupby("event_id", as_index=True).agg(
        funding_rate=("funding_rate", "first"),
        symbol=("symbol", "first"),
    )
    event_level = meta.join(path, how="inner").dropna(subset=["funding_rate", "tau_0", "tau_60"])
    if len(event_level) < 16:
        warnings.warn("有效真实事件少于 16 个，无法稳定绘制度量反应图")
        return
    event_level["abs_funding_bp"] = event_level["funding_rate"].abs() * 10000
    # Rank first makes qcut robust to tied funding rates at zero.
    event_level["dose_bin"] = pd.qcut(
        event_level["abs_funding_bp"].rank(method="first"),
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
    )

    rows = []
    for dose, part in event_level.groupby("dose_bin", observed=False):
        for horizon, column in [("tau=0", "tau_0"), ("tau=+60", "tau_60")]:
            values = part[column].astype(float)
            mean = values.mean()
            se = values.std(ddof=1) / np.sqrt(values.count()) if values.count() > 1 else np.nan
            rows.append(
                {
                    "dose_bin": str(dose),
                    "horizon": horizon,
                    "mean_bp": mean,
                    "lo95": mean - 1.96 * se if np.isfinite(se) else np.nan,
                    "hi95": mean + 1.96 * se if np.isfinite(se) else np.nan,
                    "count": int(values.count()),
                    "mean_abs_funding_bp": part["abs_funding_bp"].mean(),
                }
            )
    dose_summary = pd.DataFrame(rows)
    dose_summary.to_csv(OUTPUT_DIR / "费率剂量反应汇总.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    x = np.arange(4)
    for horizon, color in [("tau=0", "#B42318"), ("tau=+60", "#1570EF")]:
        part = dose_summary.loc[dose_summary["horizon"] == horizon].copy()
        part["order"] = part["dose_bin"].map({"Q1 lowest": 0, "Q2": 1, "Q3": 2, "Q4 highest": 3})
        part = part.sort_values("order")
        ax.errorbar(
            x,
            part["mean_bp"],
            yerr=[part["mean_bp"] - part["lo95"], part["hi95"] - part["mean_bp"]],
            marker="o",
            linewidth=2,
            capsize=4,
            color=color,
            label={"tau=0": "结算时点", "tau=+60": "结算后 60 分钟"}[horizon],
        )
    ax.axhline(0, color="#98A2B3", linewidth=0.8)
    ax.set_xticks(x, ["Q1\n最低", "Q2", "Q3", "Q4\n最高"])
    ax.set_xlabel("绝对资金费率分位数（事件层）")
    ax.set_ylabel("累计价格变化（基点）")
    ax.set_title("资金费率幅度与结算附近价格响应：剂量反应探索")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "三组事件图_费率剂量反应.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    funding, prices = read_input()
    events = make_events(funding)
    events.to_csv(OUTPUT_DIR / "事件索引.csv", index=False, encoding="utf-8-sig")

    # 如果伪事件偏移小于等于分析窗口，伪事件可能和真实事件窗口重叠。
    interval = (
        events.loc[events["group"] == "A_real_settlement"]
        .groupby("symbol")["interval_minutes"]
        .median()
    )
    for symbol, minutes in interval.items():
        if minutes / 2 <= WINDOW_MINUTES:
            warnings.warn(
                f"{symbol} 的半结算周期为 {minutes / 2:.1f} 分钟，"
                f"不大于窗口 {WINDOW_MINUTES} 分钟；请缩短窗口或重新设计伪事件。"
            )

    panel = extract_event_panel(events, prices)
    if panel.empty:
        raise RuntimeError("没有形成有效事件窗口，请检查时间戳、价格频率和数据覆盖。")
    panel.to_csv(OUTPUT_DIR / "事件面板.csv", index=False, encoding="utf-8-sig")
    summary = summarize(panel)
    summary.to_csv(OUTPUT_DIR / "事件路径汇总.csv", index=False, encoding="utf-8-sig")
    plot_dose_response(panel)

    counts = (
        panel[["event_id", "group"]].drop_duplicates().groupby("group").size()
        .rename("valid_event_count").reset_index()
    )
    counts.to_csv(OUTPUT_DIR / "事件样本量.csv", index=False, encoding="utf-8-sig")
    plot_metric(summary, "cumulative_bp", "累计价格变化（基点，事件前 -10 至 -1 分钟为基准）", "三组事件路径_累计价格变化.png")
    plot_metric(summary, "minute_return_bp", "平均逐分钟收益（基点）", "三组事件路径_单分钟收益.png")
    print(f"完成：{OUTPUT_DIR}")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
