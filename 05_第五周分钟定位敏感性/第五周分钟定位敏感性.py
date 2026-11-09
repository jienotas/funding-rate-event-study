# -*- coding: utf-8 -*-
"""第五周：事件分钟定位敏感性。

第一周的事件时间和 1 分钟价格都被统一到 ``floor('min')``，因此原始面板
中的 tau=-1/0/+1 只能描述相对事件路径，不能检验事件时间落在相邻分钟时
结论是否改变。本脚本把每个事件标记分别平移 -1、0、+1 分钟，重新从原始
1 分钟永续价格计算事件收益，并对真实/伪事件差异做 UTC 日期聚类 bootstrap。

本脚本只读取第一周已有数据，所有新结果写入本目录；不把 realized funding
解释为预测费率，也不把未跨越零的置信区间解释为因果证明。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WEEK1 = ROOT.parent / "01_第一周可行性测试"
DATA = ROOT / "data"
OUT = ROOT / "output"
SEED = 20261006
N_BOOT = 4000
MARKER_SHIFTS = (-1, 0, 1)
TAUS = (-1, 0, 1, 5, 60, 120)
BASELINE_TAUS = tuple(range(-10, 0))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_prices() -> dict[str, pd.Series]:
    """加载第一周分钟价格，保持与第一周脚本相同的重采样/短缺口规则。"""
    path = WEEK1 / "data" / "perp_1m.csv"
    prices = pd.read_csv(path)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True, format="mixed").dt.floor("min")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.dropna(subset=["timestamp", "symbol", "price"])
    prices = prices.drop_duplicates(["symbol", "timestamp"])
    result: dict[str, pd.Series] = {}
    for symbol, part in prices.groupby("symbol"):
        series = part.set_index("timestamp")["price"].sort_index()
        result[symbol] = series.resample("1min").last().ffill(limit=1)
    return result


def load_events() -> pd.DataFrame:
    events = pd.read_csv(WEEK1 / "output" / "事件索引.csv")
    # 保持与第一周完整 [-120,+120] 事件窗口相同的有效样本，避免本周的
    # 局部窗口把边界事件重新纳入而改变真实/伪事件的样本构成。
    valid_ids = pd.read_csv(
        WEEK1 / "output" / "事件面板.csv", usecols=["event_id"]
    )["event_id"].drop_duplicates()
    events = events.loc[events["event_id"].isin(valid_ids)].copy()
    events["event_time"] = pd.to_datetime(events["timestamp"], utc=True, format="mixed").dt.floor("min")
    events["funding_rate"] = pd.to_numeric(events["funding_rate"], errors="coerce")
    events["event_date"] = events["event_time"].dt.strftime("%Y-%m-%d")
    return events


def save_sample_audit(events: pd.DataFrame) -> None:
    all_events = pd.read_csv(WEEK1 / "output" / "事件索引.csv")
    excluded = all_events.loc[~all_events["event_id"].isin(events["event_id"])].copy()
    excluded["reason"] = "not in first-week complete [-120,+120] panel"
    excluded.to_csv(DATA / "第五周与第一周样本审计_排除事件.csv", index=False, encoding="utf-8-sig")


def one_event(series: pd.Series, marker: pd.Timestamp) -> dict[str, float] | None:
    """用 marker 前十分钟中位数作基准，计算指定相对时点的累计基点变化。"""
    all_taus = sorted(set(BASELINE_TAUS) | set(TAUS))
    target_times = marker + pd.to_timedelta(all_taus, unit="m")
    values = series.reindex(target_times)
    if values.isna().any() or (values <= 0).any():
        return None
    baseline = float(values.loc[marker + pd.to_timedelta(BASELINE_TAUS, unit="m")].median())
    car = 10000 * np.log(values.to_numpy(dtype=float) / baseline)
    by_tau = dict(zip(all_taus, car))
    output = {f"car_{tau:+d}": float(by_tau[tau]) for tau in TAUS}
    output["local_jump_m1_to_0"] = output["car_+0"] - output["car_-1"]
    output["local_jump_0_to_p1"] = output["car_+1"] - output["car_+0"]
    return output


def build_event_panel(events: pd.DataFrame, prices: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    skipped: list[dict] = []
    for event in events.itertuples(index=False):
        series = prices.get(event.symbol)
        if series is None:
            for shift in MARKER_SHIFTS:
                skipped.append({"event_id": event.event_id, "marker_shift_min": shift, "reason": "missing_symbol"})
            continue
        for shift in MARKER_SHIFTS:
            marker = event.event_time + pd.Timedelta(minutes=shift)
            outcome = one_event(series, marker)
            if outcome is None:
                skipped.append({"event_id": event.event_id, "marker_shift_min": shift, "reason": "incomplete_window"})
                continue
            row = {
                "event_id": int(event.event_id),
                "symbol": event.symbol,
                "event_time": event.event_time,
                "marker_time": marker,
                "marker_shift_min": int(shift),
                "group": event.group,
                "funding_rate": event.funding_rate,
                "event_date": event.event_time.strftime("%Y-%m-%d"),
            }
            row.update(outcome)
            rows.append(row)
    panel = pd.DataFrame(rows)
    skipped_frame = pd.DataFrame(skipped)
    return panel, skipped_frame


def bootstrap_difference(part: pd.DataFrame, outcome: str) -> dict:
    """按 UTC 日期重采样，保留真实和伪事件在同一日期内的相关性。"""
    real = part.loc[part["group"] == "A_real_settlement"].dropna(subset=[outcome, "event_date"])
    pseudo = part.loc[part["group"] == "B_pseudo_settlement"].dropna(subset=[outcome, "event_date"])
    dates = np.array(sorted(set(real["event_date"]) | set(pseudo["event_date"])))
    if len(dates) < 2 or real.empty or pseudo.empty:
        return {
            "real_minus_pseudo_bp": np.nan, "bootstrap_se_bp": np.nan,
            "lo95_bp": np.nan, "hi95_bp": np.nan, "mde80_bp": np.nan,
            "real_n": int(len(real)), "pseudo_n": int(len(pseudo)), "date_clusters": int(len(dates)),
        }
    observed = float(real[outcome].mean() - pseudo[outcome].mean())
    r_group = real.groupby("event_date")[outcome].agg(["sum", "count"]).reindex(dates, fill_value=0)
    p_group = pseudo.groupby("event_date")[outcome].agg(["sum", "count"]).reindex(dates, fill_value=0)
    outcome_seed = sum(ord(ch) for ch in outcome)
    rng = np.random.default_rng(SEED + outcome_seed)
    sampled = rng.integers(0, len(dates), size=(N_BOOT, len(dates)))
    r_sum = r_group["sum"].to_numpy()[sampled].sum(axis=1)
    r_n = r_group["count"].to_numpy()[sampled].sum(axis=1)
    p_sum = p_group["sum"].to_numpy()[sampled].sum(axis=1)
    p_n = p_group["count"].to_numpy()[sampled].sum(axis=1)
    draws = r_sum / r_n - p_sum / p_n
    se = float(draws.std(ddof=1))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "real_minus_pseudo_bp": observed,
        "bootstrap_se_bp": se,
        "lo95_bp": float(lo),
        "hi95_bp": float(hi),
        "mde80_bp": float(2.8 * se),
        "real_n": int(len(real)),
        "pseudo_n": int(len(pseudo)),
        "date_clusters": int(len(dates)),
    }


def summarize(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [f"car_{tau:+d}" for tau in TAUS] + ["local_jump_m1_to_0", "local_jump_0_to_p1"]
    group_rows = []
    diff_rows = []
    for shift, shift_part in panel.groupby("marker_shift_min"):
        for group, group_part in shift_part.groupby("group"):
            for outcome in outcomes:
                vals = pd.to_numeric(group_part[outcome], errors="coerce").dropna()
                group_rows.append({
                    "marker_shift_min": int(shift), "group": group, "outcome": outcome,
                    "mean_bp": float(vals.mean()) if not vals.empty else np.nan,
                    "sd_bp": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "n": int(len(vals)),
                })
        for outcome in outcomes:
            row = bootstrap_difference(shift_part, outcome)
            row.update({"marker_shift_min": int(shift), "outcome": outcome})
            row["equivalent_pm1bp"] = bool(
                np.isfinite(row["lo95_bp"]) and row["lo95_bp"] > -1 and row["hi95_bp"] < 1
            )
            diff_rows.append(row)
    return pd.DataFrame(group_rows), pd.DataFrame(diff_rows)


def compare_original(panel: pd.DataFrame) -> pd.DataFrame:
    """验证 shift=0 与第一周事件面板的 tau=-1/0/+1 是否一致。"""
    original = pd.read_csv(WEEK1 / "output" / "事件面板.csv", usecols=["event_id", "tau", "cumulative_bp"])
    original = original.loc[original["tau"].isin(TAUS)].copy()
    original["outcome"] = original["tau"].map(lambda x: f"car_{int(x):+d}")
    original = original.pivot_table(index="event_id", columns="outcome", values="cumulative_bp", aggfunc="first").reset_index()
    current = panel.loc[panel["marker_shift_min"] == 0, ["event_id"] + [f"car_{tau:+d}" for tau in TAUS]].copy()
    merged = current.merge(original, on="event_id", how="inner", suffixes=("_new", "_old"))
    rows = []
    for tau in TAUS:
        col = f"car_{tau:+d}"
        diff = merged[f"{col}_new"] - merged[f"{col}_old"]
        rows.append({"tau": int(tau), "matched_events": int(diff.notna().sum()), "max_abs_diff_bp": float(diff.abs().max()), "mean_abs_diff_bp": float(diff.abs().mean())})
    return pd.DataFrame(rows)


def plot_sensitivity(diff: pd.DataFrame) -> None:
    outcomes = ["car_-1", "car_+0", "car_+1", "car_+60"]
    labels = {"car_-1": "tau=-1", "car_+0": "tau=0", "car_+1": "tau=+1", "car_+60": "tau=+60"}
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
    colors = {"car_-1": "#667085", "car_+0": "#B42318", "car_+1": "#1570EF", "car_+60": "#7F56D9"}
    x = np.array(MARKER_SHIFTS)
    for outcome in outcomes:
        part = diff.loc[diff["outcome"] == outcome].set_index("marker_shift_min").reindex(MARKER_SHIFTS)
        ax.errorbar(
            x, part["real_minus_pseudo_bp"],
            yerr=[part["real_minus_pseudo_bp"] - part["lo95_bp"], part["hi95_bp"] - part["real_minus_pseudo_bp"]],
            marker="o", linewidth=1.8, capsize=3, color=colors[outcome], label=labels[outcome],
        )
    ax.axhline(0, color="#98A2B3", linewidth=0.8)
    ax.axhspan(-1, 1, color="#98A2B3", alpha=0.08, label="±1 bp 区间")
    ax.set_xticks(x, ["-1 分钟", "原始定位", "+1 分钟"])
    ax.set_xlabel("事件标记相对第一周时间戳的平移")
    ax.set_ylabel("真实事件 - 伪事件（基点）")
    ax.set_title("第五周：事件分钟定位敏感性（日期聚类 bootstrap）")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "第五周_分钟定位敏感性.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events()
    save_sample_audit(events)
    prices = load_prices()
    panel, skipped = build_event_panel(events, prices)
    panel.to_csv(DATA / "第五周分钟定位敏感性_事件级.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(DATA / "第五周分钟定位敏感性_跳过事件.csv", index=False, encoding="utf-8-sig")
    groups, differences = summarize(panel)
    groups.to_csv(OUT / "第五周分钟定位敏感性_分组均值.csv", index=False, encoding="utf-8-sig")
    differences.to_csv(OUT / "第五周分钟定位敏感性_真实减伪_日期聚类bootstrap.csv", index=False, encoding="utf-8-sig")
    original_check = compare_original(panel)
    original_check.to_csv(OUT / "第五周与第一周原始tau对照.csv", index=False, encoding="utf-8-sig")
    plot_sensitivity(differences)
    print("events", len(events), "panel_rows", len(panel), "skipped_rows", len(skipped))
    print("original tau check")
    print(original_check.to_string(index=False))
    print("local positioning sensitivity")
    print(differences.loc[differences["outcome"].isin(["car_+0", "car_+1", "local_jump_m1_to_0"])].to_string(index=False))


if __name__ == "__main__":
    main()
