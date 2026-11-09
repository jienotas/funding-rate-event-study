# -*- coding: utf-8 -*-
"""Week 11: clock-phase and same-minute-slot audit.

This stage does not create a causal control group. It asks whether the Week 10
lead contrast is concentrated in clock phase or funding-hour slots. The primary
comparison is the within-event 0-minute jump minus the -60-minute jump, because
both anchors occur at minute 00 and therefore share the minute-of-hour phase.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
EVENTS_PATH = PROJECT / "01_第一周可行性测试" / "output" / "事件索引.csv"
BARS_PATH = PROJECT / "06_第六周流动性机制" / "data" / "binance_futures_5m_liquidity.csv"
DATA = ROOT / "data"
OUT = ROOT / "output"
SEED = 20261007
N_BOOT = 4000
MAIN_OUTCOMES = ["jump_quote_volume_log", "jump_range_bp", "jump_amihud_log1p"]
OUTCOMES = {
    "jump_quote_volume_log": "log成交额变化",
    "jump_range_bp": "高低价振幅变化_bp",
    "jump_amihud_log1p": "log1p成交额归一化绝对收益代理变化",
    "jump_taker_imbalance": "主动买入不平衡变化",
}
BAR_COLUMNS = ("open", "high", "low", "close", "quote_volume", "taker_buy_quote")


def load_bars() -> pd.DataFrame:
    frame = pd.read_csv(BARS_PATH, usecols=["symbol", "timestamp", *BAR_COLUMNS])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for col in BAR_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["symbol", "timestamp"])
    if frame.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("5分钟K线存在重复 symbol-timestamp 键，停止分析。")
    return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def load_events() -> pd.DataFrame:
    frame = pd.read_csv(EVENTS_PATH)
    frame = frame.loc[frame["group"].eq("A_real_settlement")].copy()
    frame["funding_time"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["symbol", "funding_time"])
    if frame.duplicated(["symbol", "funding_time"]).any():
        raise ValueError("真实结算事件存在重复 symbol-timestamp 键，停止分析。")
    frame["funding_event_key"] = (
        frame["symbol"].astype(str) + "|" + frame["funding_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    frame["event_date"] = frame["funding_time"].dt.strftime("%Y-%m-%d")
    frame["funding_hour_utc"] = frame["funding_time"].dt.hour
    return frame[["symbol", "funding_time", "funding_event_key", "event_date", "funding_hour_utc"]]


def compute_jumps(bars: pd.DataFrame) -> pd.DataFrame:
    """Create every adjacent 5-minute local jump for phase diagnostics."""
    pre = bars.rename(columns={"timestamp": "pre_time", **{c: f"pre_{c}" for c in BAR_COLUMNS}})
    post = bars.rename(columns={"timestamp": "post_time", **{c: f"post_{c}" for c in BAR_COLUMNS}})
    pre["post_time"] = pre["pre_time"] + pd.Timedelta(minutes=5)
    joined = post.merge(pre, on=["symbol", "post_time"], how="left", validate="one_to_one")
    valid_price_pre = joined[["pre_open", "pre_close", "pre_high", "pre_low"]].gt(0).all(axis=1)
    valid_price_post = joined[["post_open", "post_close", "post_high", "post_low"]].gt(0).all(axis=1)
    valid_quote_pre = joined["pre_quote_volume"].gt(0)
    valid_quote_post = joined["post_quote_volume"].gt(0)
    pre_ret = np.where(valid_price_pre, np.abs(np.log(joined["pre_close"] / joined["pre_open"])) * 10000, np.nan)
    post_ret = np.where(valid_price_post, np.abs(np.log(joined["post_close"] / joined["post_open"])) * 10000, np.nan)
    pre_range = np.where(valid_price_pre, np.log(joined["pre_high"] / joined["pre_low"]) * 10000, np.nan)
    post_range = np.where(valid_price_post, np.log(joined["post_high"] / joined["post_low"]) * 10000, np.nan)
    pre_qlog = np.where(valid_quote_pre, np.log(joined["pre_quote_volume"]), np.nan)
    post_qlog = np.where(valid_quote_post, np.log(joined["post_quote_volume"]), np.nan)
    pre_amihud = np.where(valid_quote_pre, pre_ret / (joined["pre_quote_volume"] / 1_000_000), np.nan)
    post_amihud = np.where(valid_quote_post, post_ret / (joined["post_quote_volume"] / 1_000_000), np.nan)
    pre_taker = np.where(valid_quote_pre, 2 * joined["pre_taker_buy_quote"] / joined["pre_quote_volume"] - 1, np.nan)
    post_taker = np.where(valid_quote_post, 2 * joined["post_taker_buy_quote"] / joined["post_quote_volume"] - 1, np.nan)
    out = pd.DataFrame({
        "symbol": joined["symbol"], "timestamp": joined["post_time"],
        "jump_quote_volume_log": post_qlog - pre_qlog,
        "jump_range_bp": post_range - pre_range,
        "jump_amihud_log1p": np.log1p(np.clip(post_amihud, 0, None)) - np.log1p(np.clip(pre_amihud, 0, None)),
        "jump_taker_imbalance": post_taker - pre_taker,
    })
    out = out.dropna(subset=["timestamp"])
    out["event_date"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    out["utc_hour"] = out["timestamp"].dt.hour
    out["minute_of_hour"] = out["timestamp"].dt.minute
    return out


def bootstrap_mean(frame: pd.DataFrame, value_col: str, seed: int, block_days: int | None = None) -> tuple[float, float]:
    grouped = frame.groupby("event_date")[value_col].agg(["sum", "count"])
    grouped = grouped.loc[grouped["count"].gt(0)]
    if grouped.empty:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    if block_days is None:
        ids = rng.integers(0, len(grouped), size=(N_BOOT, len(grouped)))
        sums = grouped["sum"].to_numpy()[ids].sum(axis=1)
        counts = grouped["count"].to_numpy()[ids].sum(axis=1)
    else:
        calendar = pd.date_range(pd.Timestamp(grouped.index.min()), pd.Timestamp(grouped.index.max()), freq="D").strftime("%Y-%m-%d")
        grouped = grouped.reindex(calendar, fill_value=0)
        n_days = len(grouped)
        n_blocks = math.ceil(n_days / block_days)
        starts = rng.integers(0, n_days, size=(N_BOOT, n_blocks))
        offsets = np.arange(block_days)
        days = (starts[:, :, None] + offsets[None, None, :]) % n_days
        days = days.reshape(N_BOOT, -1)[:, :n_days]
        sums = grouped["sum"].to_numpy()[days].sum(axis=1)
        counts = grouped["count"].to_numpy()[days].sum(axis=1)
    valid = counts > 0
    draws = sums[valid] / counts[valid]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def event_panel(jumps: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    anchors = []
    for offset in (-60, -45, -30, -15, 0):
        part = events.copy()
        part["anchor_offset_min"] = offset
        part["anchor_time"] = part["funding_time"] + pd.to_timedelta(offset, unit="m")
        anchors.append(part)
    anchors = pd.concat(anchors, ignore_index=True)
    jump_values = jumps.drop(columns=["event_date"], errors="ignore")
    panel = anchors.merge(jump_values, left_on=["symbol", "anchor_time"], right_on=["symbol", "timestamp"], how="left", validate="one_to_one")
    panel["anchor_minute_of_hour"] = panel["anchor_time"].dt.minute
    panel["anchor_hour_utc"] = panel["anchor_time"].dt.hour
    return panel


def phase_summary(jumps: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    event_keys = events[["symbol", "funding_time"]].rename(columns={"funding_time": "timestamp"})
    tagged = jumps.merge(event_keys.assign(is_event=True), on=["symbol", "timestamp"], how="left")
    tagged["is_event"] = tagged["is_event"].fillna(False).astype(bool)
    rows = []
    for outcome in OUTCOMES:
        non_event = tagged.loc[tagged["is_event"].eq(False)].dropna(subset=[outcome])
        for minute in sorted(tagged["minute_of_hour"].dropna().unique()):
            part = non_event.loc[non_event["minute_of_hour"].eq(minute)]
            if part.empty:
                continue
            lo, hi = bootstrap_mean(part, outcome, SEED + int(minute) + list(OUTCOMES).index(outcome) * 100)
            rows.append({
                "outcome": outcome, "outcome_label": OUTCOMES[outcome], "minute_of_hour": int(minute),
                "mean_jump_non_event": float(part[outcome].mean()), "date_bootstrap_ci95_lo": lo,
                "date_bootstrap_ci95_hi": hi, "n": len(part), "date_clusters": part["event_date"].nunique(),
            })
    return pd.DataFrame(rows)


def paired_contrasts(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        wide = panel.pivot(index=["funding_event_key", "symbol", "event_date", "funding_hour_utc"], columns="anchor_offset_min", values=outcome).reset_index()
        for offset in (-60, -45, -30, -15):
            data = wide.dropna(subset=[0, offset]).copy()
            data["contrast"] = data[0] - data[offset]
            if data.empty:
                continue
            model = sm.OLS(data["contrast"], np.ones((len(data), 1))).fit(
                cov_type="cluster", cov_kwds={"groups": data["event_date"], "use_correction": True}, use_t=True
            )
            ci = np.asarray(model.conf_int(alpha=0.05))[0]
            boot_lo, boot_hi = bootstrap_mean(data.rename(columns={"contrast": "bootstrap_value"}), "bootstrap_value", SEED + 1000 + outcome_index * 10 + abs(offset))
            block_lo, block_hi = bootstrap_mean(data.rename(columns={"contrast": "bootstrap_value"}), "bootstrap_value", SEED + 2000 + outcome_index * 10 + abs(offset), block_days=7)
            block14_lo, block14_hi = bootstrap_mean(data.rename(columns={"contrast": "bootstrap_value"}), "bootstrap_value", SEED + 3000 + outcome_index * 10 + abs(offset), block_days=14)
            rows.append({
                "outcome": outcome, "outcome_label": OUTCOMES[outcome], "lead_offset_min": offset,
                "endpoint_role": "primary" if outcome == "jump_quote_volume_log" else ("secondary" if outcome in MAIN_OUTCOMES else "exploratory"),
                "n_event_asset": len(data), "date_clusters": data["event_date"].nunique(),
                "mean_real_minus_lead": float(data["contrast"].mean()), "cluster_ci95_lo": float(ci[0]),
                "cluster_ci95_hi": float(ci[1]), "cluster_p_value": float(np.asarray(model.pvalues)[0]),
                "date_bootstrap_ci95_lo": boot_lo, "date_bootstrap_ci95_hi": boot_hi,
                "moving_7d_block_ci95_lo": block_lo, "moving_7d_block_ci95_hi": block_hi,
                "moving_14d_block_ci95_lo": block14_lo, "moving_14d_block_ci95_hi": block14_hi,
            })
    result = pd.DataFrame(rows)
    result["bh_q_value_12_main_lead_family"] = np.nan
    main = result["outcome"].isin(MAIN_OUTCOMES)
    if main.any():
        result.loc[main, "bh_q_value_12_main_lead_family"] = multipletests(result.loc[main, "cluster_p_value"], method="fdr_bh")[1]
    return result


def same_minute_slot(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        wide = panel.pivot(index=["funding_event_key", "symbol", "event_date", "funding_hour_utc"], columns="anchor_offset_min", values=outcome).reset_index()
        data = wide.dropna(subset=[0, -60]).copy()
        data["contrast"] = data[0] - data[-60]
        for symbol in sorted(data["symbol"].unique()):
            for hour in sorted(data["funding_hour_utc"].unique()):
                part = data.loc[data["symbol"].eq(symbol) & data["funding_hour_utc"].eq(hour)].dropna(subset=["contrast"])
                if part.empty:
                    continue
                model = sm.OLS(part["contrast"], np.ones((len(part), 1))).fit(
                    cov_type="cluster", cov_kwds={"groups": part["event_date"], "use_correction": True}, use_t=True
                )
                ci = np.asarray(model.conf_int(alpha=0.05))[0]
                rows.append({
                    "outcome": outcome, "outcome_label": OUTCOMES[outcome], "symbol": symbol,
                    "funding_hour_utc": int(hour), "n": len(part), "date_clusters": part["event_date"].nunique(),
                    "mean_real_minus_minus60": float(model.params.iloc[0]), "cluster_ci95_lo": float(ci[0]), "cluster_ci95_hi": float(ci[1]),
                    "cluster_p_value": float(model.pvalues.iloc[0]),
                })
    return pd.DataFrame(rows)


def quality(jumps: pd.DataFrame, events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"metric": "adjacent_5m_jump_pairs", "value": int(len(jumps))},
        {"metric": "event_count", "value": int(len(events))},
        {"metric": "event_anchor_rows", "value": int(len(panel))},
        {"metric": "event_anchor_rows_with_jump", "value": int(panel["jump_quote_volume_log"].notna().sum())},
        {"metric": "event_anchor_missing_jump", "value": int(panel["jump_quote_volume_log"].isna().sum())},
        {"metric": "event_dates", "value": int(events["event_date"].nunique())},
        {"metric": "jump_min_utc", "value": jumps["timestamp"].min().isoformat()},
        {"metric": "jump_max_utc", "value": jumps["timestamp"].max().isoformat()},
        {"metric": "non_event_minute_phases", "value": int(jumps["minute_of_hour"].nunique())},
    ])


def write_report(events: pd.DataFrame, jumps: pd.DataFrame, panel: pd.DataFrame, phase: pd.DataFrame, paired: pd.DataFrame, slots: pd.DataFrame, qc: pd.DataFrame, duration: float) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "## Material Passport", "", "- Origin Skill: experiment-agent", "- Origin Mode: run", f"- Origin Date: {now}",
        "- Verification Status: VERIFIED", "- Version Label: exp_result_v1", "", "## Experiment Result", "",
        "- **ID**: week11-clock-phase-slot-audit-20261007", "- **Type**: analysis", "- **Status**: completed",
        '- **Command**: `py ".\\第十一周时钟相位与同槽位审计.py"`', f"- **Working Directory**: `{ROOT}`", f"- **Duration**: {duration:.2f} seconds", "- **Exit Code**: 0", "",
        "### Purpose and estimand", "",
        "- 本阶段检验第十周差异是否集中在分钟相位、UTC 结算槽位或资产，而不是建立未处理因果对照。",
        "- 核心相位匹配诊断为同一事件的 `0` 分钟跳变减 `-60` 分钟跳变；两个锚点均位于 minute 00，因此比 `0` 减 `-45/-30/-15` 更少混入分钟相位差。该比较是看到第十周曲线后的敏感性分析，不是预注册确认性检验。",
        "- 该对照仍改变 UTC 小时，且结算事件按固定时刻发生；因此结果只能作为时钟混杂审计和时点关联证据。", "",
        "### Same-minute phase audit", "", "| 指标 | lead | 平均差异 | 日期聚类95% CI | 7日块95% CI | 14日块95% CI | p | BH q(12主端点) |", "|------|-----:|---------:|---------------:|-------------:|---------------:|--:|----------------:|",
    ]
    for row in paired.itertuples(index=False):
        q = row.bh_q_value_12_main_lead_family
        lines.append(f"| {row.outcome_label} | {row.lead_offset_min} | {row.mean_real_minus_lead:.6g} | [{row.cluster_ci95_lo:.6g}, {row.cluster_ci95_hi:.6g}] | [{row.moving_7d_block_ci95_lo:.6g}, {row.moving_7d_block_ci95_hi:.6g}] | [{row.moving_14d_block_ci95_lo:.6g}, {row.moving_14d_block_ci95_hi:.6g}] | {row.cluster_p_value:.6g} | {'NA' if pd.isna(q) else f'{q:.6g}'} |")
    lines.extend(["", "### Slot-specific 0 minus -60", "", "| 指标 | 资产 | UTC小时 | 平均差异 | 日期聚类95% CI | p |", "|------|------|--------:|---------:|---------------:|--:|"])
    for row in slots.itertuples(index=False):
        lines.append(f"| {row.outcome_label} | {row.symbol} | {row.funding_hour_utc} | {row.mean_real_minus_minus60:.6g} | [{row.cluster_ci95_lo:.6g}, {row.cluster_ci95_hi:.6g}] | {row.cluster_p_value:.6g} |")
    lines.extend(["", "### Interpretation", "", "- 若 `0−(-60)` 比 `0−(-45/-30/-15)` 稳定得多，说明第十周主差异很大一部分可能来自 minute-of-hour 相位，而非逐步接近结算的反应。",
                  "- UTC 槽位之间方向或大小不一致时，不能用一个 pooled 均值概括所有结算。",
                  "- 非事件跳变的分钟相位均值只用于描述系统性 K 线时钟结构；它不是未处理同槽位因果对照，因为 exact hour×minute 的结算单元没有同刻未结算观测。",
                  "- 四个 lead 的逐点配对比较按三个主要端点构成12项BH家族；主动买入不平衡为探索性，不进入此家族。`0−(-60)`虽为相位匹配诊断，但由于它是在观察第十周曲线后提出，不能视作独立确认。",
                  "- 所有结果仍基于 5 分钟 OHLCV 代理；不能称为盘口价差、订单簿深度或执行成本。", "",
                  "### Non-event phase summary", "", "| 指标 | minute | 非事件平均跳变 | 日期聚类bootstrap CI | N |", "|------|-------:|---------------:|----------------------:|--:|"])
    for row in phase.loc[phase["outcome"].isin(MAIN_OUTCOMES)].itertuples(index=False):
        lines.append(f"| {row.outcome_label} | {row.minute_of_hour} | {row.mean_jump_non_event:.6g} | [{row.date_bootstrap_ci95_lo:.6g}, {row.date_bootstrap_ci95_hi:.6g}] | {row.n} |")
    lines.extend(["", "### Data Quality", "", "| Metric | Value |", "|--------|------:|"])
    for row in qc.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.value} |")
    lines.extend(["", "### Boundary", "", "- 这是时钟相位与槽位审计，不是差分中的差分、随机实验或因果识别。", "- `0`、`8`、`16` UTC 结算几乎固定发生在 minute 00，因而不能从本数据单独分离结算机制效应与 exact-slot 时钟效应。", "- 下一步若要提高识别，需要外部交易所/不同结算时刻、明确未处理同槽位，或可验证的事件时间变化；扩大同一固定时刻样本不会自动解决该问题。", "", "### Output Files", "", "- `data/第十一周全样本5分钟跳变.csv`", "- `data/第十一周事件锚点面板.csv`", "- `output/第十一周非事件分钟相位.csv`", "- `output/第十一周同分钟主对照.csv`", "- `output/第十一周槽位分层.csv`", "- `output/第十一周数据质量摘要.csv`"])
    (OUT / "第十一周实验结果与解释.md").write_text("\n".join(lines), encoding="utf-8")


def save_phase_figure(phase: pd.DataFrame) -> None:
    labels = ["jump_quote_volume_log", "jump_range_bp"]
    titles = ["Log quote-volume local jump", "Range local jump (bp)"]
    colors = ["#177E89", "#D95D39"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for ax, outcome, title, color in zip(axes, labels, titles, colors):
        part = phase.loc[phase["outcome"].eq(outcome)].sort_values("minute_of_hour")
        x = part["minute_of_hour"].to_numpy()
        y = part["mean_jump_non_event"].to_numpy()
        lo = part["date_bootstrap_ci95_lo"].to_numpy()
        hi = part["date_bootstrap_ci95_hi"].to_numpy()
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="o-", color=color, ecolor="#50555A", capsize=3, linewidth=1.8)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(x, [f":{int(v):02d}" for v in x])
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Mean adjacent-bar jump")
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("UTC minute phase of the 5-minute bar")
    fig.suptitle("Non-settlement 5-minute jumps vary by clock phase", fontsize=13)
    fig.savefig(OUT / "第十一周分钟相位时钟效应.png", dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events()
    bars = load_bars()
    jumps = compute_jumps(bars)
    panel = event_panel(jumps, events)
    phase = phase_summary(jumps, events)
    paired = paired_contrasts(panel)
    slots = same_minute_slot(panel)
    qc = quality(jumps, events, panel)
    jumps.to_csv(DATA / "第十一周全样本5分钟跳变.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(DATA / "第十一周事件锚点面板.csv", index=False, encoding="utf-8-sig")
    phase.to_csv(OUT / "第十一周非事件分钟相位.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT / "第十一周同分钟主对照.csv", index=False, encoding="utf-8-sig")
    slots.to_csv(OUT / "第十一周槽位分层.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(OUT / "第十一周数据质量摘要.csv", index=False, encoding="utf-8-sig")
    duration = time.perf_counter() - started
    write_report(events, jumps, panel, phase, paired, slots, qc, duration)
    save_phase_figure(phase)
    print(f"events={len(events)} jumps={len(jumps)} panel={len(panel)} matched={int(panel['jump_quote_volume_log'].notna().sum())}")
    print("paired")
    print(paired.to_string(index=False))
    print("quality")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
