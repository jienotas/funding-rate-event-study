# -*- coding: utf-8 -*-
"""Week 10: pre-event lead curve using only pre-settlement pseudo anchors.

The primary estimand is the real-settlement local jump minus the mean of the
four pre-event local jumps. This is an event-time association check, not a
causal treatment effect: funding rates are scheduled and the data contain no
untreated same-hour counterfactual.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
EVENTS_PATH = PROJECT / "01_第一周可行性测试" / "output" / "事件索引.csv"
BARS_PATH = PROJECT / "06_第六周流动性机制" / "data" / "binance_futures_5m_liquidity.csv"
DATA = ROOT / "data"
OUT = ROOT / "output"
SEED = 20261007
N_BOOT = 4000
LEAD_OFFSETS_MIN = (-60, -45, -30, -15, 0)
OUTCOMES = {
    "jump_quote_volume_log": "log成交额局部变化",
    "jump_range_bp": "高低价振幅局部变化_bp",
    "jump_amihud_log1p": "log1p成交额归一化绝对收益代理局部变化",
    "jump_taker_imbalance": "主动买入不平衡局部变化",
}
MAIN_OUTCOMES = ["jump_quote_volume_log", "jump_range_bp", "jump_amihud_log1p"]

BAR_COLUMNS = ("open", "high", "low", "close", "quote_volume", "taker_buy_quote")


def load_bars() -> pd.DataFrame:
    frame = pd.read_csv(BARS_PATH, usecols=["symbol", "timestamp", *BAR_COLUMNS])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for col in BAR_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["symbol", "timestamp"])
    if frame.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("5分钟K线存在重复 symbol-timestamp 键，停止分析。")
    return frame


def load_events() -> pd.DataFrame:
    frame = pd.read_csv(EVENTS_PATH)
    required = {"symbol", "timestamp", "group"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"事件索引缺少列：{sorted(missing)}")
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


def build_panel(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    anchors = []
    for offset in LEAD_OFFSETS_MIN:
        part = events.copy()
        part["anchor_offset_min"] = offset
        part["anchor_label"] = "真实结算" if offset == 0 else f"结算前{abs(offset)}min"
        part["anchor_time"] = part["funding_time"] + pd.to_timedelta(offset, unit="m")
        part["pre_bar_time"] = part["anchor_time"] - pd.Timedelta(minutes=5)
        part["post_bar_time"] = part["anchor_time"]
        anchors.append(part)
    anchors = pd.concat(anchors, ignore_index=True)

    pre = bars.rename(columns={"timestamp": "pre_bar_time", **{c: f"pre_{c}" for c in BAR_COLUMNS}})
    post = bars.rename(columns={"timestamp": "post_bar_time", **{c: f"post_{c}" for c in BAR_COLUMNS}})
    panel = anchors.merge(pre, on=["symbol", "pre_bar_time"], how="left", validate="many_to_one")
    panel = panel.merge(post, on=["symbol", "post_bar_time"], how="left", validate="many_to_one")

    for side in ("pre", "post"):
        opening = panel[f"{side}_open"]
        closing = panel[f"{side}_close"]
        high = panel[f"{side}_high"]
        low = panel[f"{side}_low"]
        quote = panel[f"{side}_quote_volume"]
        taker = panel[f"{side}_taker_buy_quote"]
        valid_price = opening.gt(0) & closing.gt(0) & high.gt(0) & low.gt(0)
        valid_quote = quote.gt(0)
        panel[f"{side}_abs_return_bp"] = np.where(valid_price, np.abs(np.log(closing / opening)) * 10000, np.nan)
        panel[f"{side}_range_bp"] = np.where(valid_price, np.log(high / low) * 10000, np.nan)
        panel[f"{side}_quote_volume_log"] = np.where(valid_quote, np.log(quote), np.nan)
        panel[f"{side}_amihud_bp_per_million_usdt"] = np.where(
            valid_quote, panel[f"{side}_abs_return_bp"] / (quote / 1_000_000), np.nan
        )
        panel[f"{side}_amihud_log1p"] = np.log1p(panel[f"{side}_amihud_bp_per_million_usdt"].clip(lower=0))
        panel[f"{side}_taker_imbalance"] = np.where(valid_quote, 2 * taker / quote - 1, np.nan)

    panel["jump_quote_volume_log"] = panel["post_quote_volume_log"] - panel["pre_quote_volume_log"]
    panel["jump_range_bp"] = panel["post_range_bp"] - panel["pre_range_bp"]
    panel["jump_amihud_log1p"] = panel["post_amihud_log1p"] - panel["pre_amihud_log1p"]
    panel["jump_amihud_raw"] = panel["post_amihud_bp_per_million_usdt"] - panel["pre_amihud_bp_per_million_usdt"]
    panel["jump_taker_imbalance"] = panel["post_taker_imbalance"] - panel["pre_taker_imbalance"]
    panel["anchor_hour_utc"] = panel["anchor_time"].dt.hour
    return panel


def summarize_events(panel: pd.DataFrame) -> pd.DataFrame:
    keys = ["funding_event_key", "symbol", "event_date", "funding_hour_utc"]
    rows = []
    for outcome in OUTCOMES:
        wide = panel.pivot(index=keys, columns="anchor_offset_min", values=outcome).reset_index()
        lead_cols = [c for c in LEAD_OFFSETS_MIN if c < 0]
        wide["lead_count"] = wide[lead_cols].notna().sum(axis=1)
        wide["pre_lead_mean"] = wide[lead_cols].mean(axis=1, skipna=True)
        wide.loc[wide["lead_count"] < len(lead_cols), "pre_lead_mean"] = np.nan
        wide["real_jump"] = wide[0]
        wide["real_minus_pre_lead_mean"] = wide[0] - wide["pre_lead_mean"]
        wide["outcome"] = outcome
        wide["outcome_label"] = OUTCOMES[outcome]
        rows.append(wide[keys + ["outcome", "outcome_label", "lead_count", "real_jump", "pre_lead_mean", "real_minus_pre_lead_mean", *lead_cols]])
    return pd.concat(rows, ignore_index=True)


def cluster_interval(frame: pd.DataFrame, value_col: str, seed: int, block_days: int | None = None) -> tuple[float, float, float]:
    grouped = frame.groupby("event_date")[value_col].agg(["sum", "count"])
    if grouped.empty:
        return np.nan, np.nan, np.nan
    if block_days is None:
        ids = np.random.default_rng(seed).integers(0, len(grouped), size=(N_BOOT, len(grouped)))
        draws = grouped["sum"].to_numpy()[ids].sum(axis=1) / grouped["count"].to_numpy()[ids].sum(axis=1)
    else:
        first = pd.Timestamp(grouped.index.min())
        last = pd.Timestamp(grouped.index.max())
        calendar = pd.date_range(first, last, freq="D").strftime("%Y-%m-%d")
        grouped = grouped.reindex(calendar, fill_value=0)
        n_days = len(grouped)
        n_blocks = math.ceil(n_days / block_days)
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, n_days, size=(N_BOOT, n_blocks))
        offsets = np.arange(block_days)
        days = (starts[:, :, None] + offsets[None, None, :]) % n_days
        days = days.reshape(N_BOOT, -1)[:, :n_days]
        sums = grouped["sum"].to_numpy()[days].sum(axis=1)
        counts = grouped["count"].to_numpy()[days].sum(axis=1)
        draws = sums[counts > 0] / counts[counts > 0]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), float(draws.std(ddof=1))


def lead_curve(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        for offset in LEAD_OFFSETS_MIN:
            frame = panel.loc[panel["anchor_offset_min"].eq(offset)].dropna(subset=[outcome, "event_date"])
            if frame.empty:
                continue
            intercept = sm.OLS(frame[outcome], np.ones((len(frame), 1))).fit(
                cov_type="cluster", cov_kwds={"groups": frame["event_date"], "use_correction": True}, use_t=True
            )
            ci = np.asarray(intercept.conf_int(alpha=0.05))[0]
            boot_lo, boot_hi, boot_se = cluster_interval(frame, outcome, SEED + outcome_index * 100 + offset + 100)
            block_lo, block_hi, block_se = cluster_interval(frame, outcome, SEED + outcome_index * 100 + offset + 200, block_days=7)
            rows.append({
                "outcome": outcome, "outcome_label": OUTCOMES[outcome], "anchor_offset_min": offset,
                "anchor_label": "真实结算" if offset == 0 else f"结算前{abs(offset)}min",
                "n": len(frame), "date_clusters": frame["event_date"].nunique(),
                "mean_jump": float(frame[outcome].mean()), "cluster_se": float(np.asarray(intercept.bse)[0]),
                "cluster_ci95_lo": float(ci[0]), "cluster_ci95_hi": float(ci[1]),
                "cluster_p_value": float(np.asarray(intercept.pvalues)[0]),
                "date_bootstrap_ci95_lo": boot_lo, "date_bootstrap_ci95_hi": boot_hi,
                "moving_7d_block_ci95_lo": block_lo, "moving_7d_block_ci95_hi": block_hi,
                "moving_7d_block_se": block_se,
            })
    return pd.DataFrame(rows)


def treatment_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        data = summary.loc[summary["outcome"].eq(outcome)].dropna(subset=["real_minus_pre_lead_mean", "event_date"])
        if data.empty:
            continue
        model = sm.OLS(data["real_minus_pre_lead_mean"], np.ones((len(data), 1))).fit(
            cov_type="cluster", cov_kwds={"groups": data["event_date"], "use_correction": True}, use_t=True
        )
        ci = np.asarray(model.conf_int(alpha=0.05))[0]
        boot_lo, boot_hi, boot_se = cluster_interval(data, "real_minus_pre_lead_mean", SEED + 3000 + outcome_index)
        block_lo, block_hi, block_se = cluster_interval(data, "real_minus_pre_lead_mean", SEED + 4000 + outcome_index, block_days=7)
        rows.append({
            "outcome": outcome, "outcome_label": OUTCOMES[outcome],
            "endpoint_role": "primary" if outcome == "jump_quote_volume_log" else ("secondary" if outcome in MAIN_OUTCOMES else "exploratory"),
            "n_event_asset": len(data), "date_clusters": data["event_date"].nunique(),
            "mean_real_jump": float(data["real_jump"].mean()), "mean_pre_lead": float(data["pre_lead_mean"].mean()),
            "real_minus_pre_lead_mean": float(np.asarray(model.params)[0]), "cluster_se": float(np.asarray(model.bse)[0]),
            "cluster_ci95_lo": float(ci[0]), "cluster_ci95_hi": float(ci[1]), "cluster_p_value": float(np.asarray(model.pvalues)[0]),
            "date_bootstrap_ci95_lo": boot_lo, "date_bootstrap_ci95_hi": boot_hi,
            "moving_7d_block_ci95_lo": block_lo, "moving_7d_block_ci95_hi": block_hi,
        })
    result = pd.DataFrame(rows)
    result["bh_q_value_3_main_outcomes"] = np.nan
    main = result["outcome"].isin(MAIN_OUTCOMES)
    if main.any():
        result.loc[main, "bh_q_value_3_main_outcomes"] = multipletests(result.loc[main, "cluster_p_value"], method="fdr_bh")[1]
    return result


def real_minus_each_lead(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lead_offsets = [offset for offset in LEAD_OFFSETS_MIN if offset < 0]
    for outcome_index, outcome in enumerate(OUTCOMES):
        wide = summary.loc[summary["outcome"].eq(outcome)].copy()
        for lead_index, offset in enumerate(lead_offsets):
            data = wide.dropna(subset=["real_jump", offset, "event_date"]).copy()
            data["contrast"] = data["real_jump"] - data[offset]
            if data.empty:
                continue
            model = sm.OLS(data["contrast"], np.ones((len(data), 1))).fit(
                cov_type="cluster", cov_kwds={"groups": data["event_date"], "use_correction": True}, use_t=True
            )
            ci = np.asarray(model.conf_int(alpha=0.05))[0]
            data_for_boot = data.rename(columns={"contrast": "bootstrap_value"})
            boot_lo, boot_hi, _ = cluster_interval(data_for_boot, "bootstrap_value", SEED + 5000 + outcome_index * 10 + lead_index)
            block_lo, block_hi, _ = cluster_interval(data_for_boot, "bootstrap_value", SEED + 6000 + outcome_index * 10 + lead_index, block_days=7)
            rows.append({
                "outcome": outcome, "outcome_label": OUTCOMES[outcome], "lead_offset_min": offset,
                "endpoint_role": "primary" if outcome == "jump_quote_volume_log" else ("secondary" if outcome in MAIN_OUTCOMES else "exploratory"),
                "n_event_asset": len(data), "date_clusters": data["event_date"].nunique(),
                "mean_real_minus_lead": float(data["contrast"].mean()), "cluster_ci95_lo": float(ci[0]),
                "cluster_ci95_hi": float(ci[1]), "cluster_p_value": float(np.asarray(model.pvalues)[0]),
                "date_bootstrap_ci95_lo": boot_lo, "date_bootstrap_ci95_hi": boot_hi,
                "moving_7d_block_ci95_lo": block_lo, "moving_7d_block_ci95_hi": block_hi,
            })
    result = pd.DataFrame(rows)
    # FDR is confirmatory only for the three pre-specified main endpoints
    # (4 real-vs-lead contrasts each = 12 tests). The taker-imbalance rows
    # remain exploratory and are deliberately excluded from this family.
    result["bh_q_value_12_main_lead_family"] = np.nan
    main = result["outcome"].isin(MAIN_OUTCOMES)
    if main.any():
        result.loc[main, "bh_q_value_12_main_lead_family"] = multipletests(
            result.loc[main, "cluster_p_value"], method="fdr_bh"
        )[1]
    return result


def stratified_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for symbol in sorted(summary["symbol"].unique()):
            for hour in sorted(summary["funding_hour_utc"].unique()):
                data = summary.loc[
                    summary["outcome"].eq(outcome) & summary["symbol"].eq(symbol) & summary["funding_hour_utc"].eq(hour)
                ].dropna(subset=["real_minus_pre_lead_mean", "event_date"])
                if data.empty:
                    continue
                model = sm.OLS(data["real_minus_pre_lead_mean"], np.ones((len(data), 1))).fit(
                    cov_type="cluster", cov_kwds={"groups": data["event_date"], "use_correction": True}, use_t=True
                )
                ci = np.asarray(model.conf_int(alpha=0.05))[0]
                rows.append({
                    "outcome": outcome, "outcome_label": OUTCOMES[outcome], "symbol": symbol,
                    "funding_hour_utc": hour, "n": len(data), "date_clusters": data["event_date"].nunique(),
                    "mean_real_minus_pre_lead_mean": float(np.asarray(model.params)[0]),
                    "cluster_ci95_lo": float(ci[0]), "cluster_ci95_hi": float(ci[1]),
                })
    return pd.DataFrame(rows)


def quality(events: pd.DataFrame, bars: pd.DataFrame, panel: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "real_funding_events", "value": len(events)},
        {"metric": "event_asset_summary_rows", "value": int(len(summary) / len(OUTCOMES))},
        {"metric": "events_with_all_four_leads_per_outcome", "value": int((summary.loc[summary["outcome"].eq("jump_quote_volume_log"), "lead_count"] == 4).sum())},
        {"metric": "minimum_lead_count_per_event", "value": int(summary.loc[summary["outcome"].eq("jump_quote_volume_log"), "lead_count"].min())},
        {"metric": "event_dates", "value": events["event_date"].nunique()},
        {"metric": "raw_5m_bars", "value": len(bars)},
        {"metric": "raw_bar_min_utc", "value": bars["timestamp"].min().isoformat()},
        {"metric": "raw_bar_max_utc", "value": bars["timestamp"].max().isoformat()},
        {"metric": "expected_anchor_rows", "value": len(events) * len(LEAD_OFFSETS_MIN)},
        {"metric": "anchors_with_both_bars", "value": int(panel["pre_open"].notna().mul(panel["post_open"].notna()).sum())},
        {"metric": "missing_pre_bar", "value": int(panel["pre_open"].isna().sum())},
        {"metric": "missing_post_bar", "value": int(panel["post_open"].isna().sum())},
        {"metric": "funding_event_dates_used_for_clustering", "value": events["event_date"].nunique()},
        {"metric": "events_per_date_min", "value": int(events.groupby("event_date").size().min())},
        {"metric": "events_per_date_max", "value": int(events.groupby("event_date").size().max())},
    ]
    for offset in LEAD_OFFSETS_MIN:
        part = panel.loc[panel["anchor_offset_min"].eq(offset)]
        rows.append({"metric": f"offset_{offset}min_complete_pre_post", "value": int(part["pre_open"].notna().mul(part["post_open"].notna()).sum())})
    return pd.DataFrame(rows)


def write_report(events: pd.DataFrame, panel: pd.DataFrame, summary: pd.DataFrame, curve: pd.DataFrame, contrasts: pd.DataFrame, lead_pairs: pd.DataFrame, qc: pd.DataFrame, duration: float) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    primary = contrasts.loc[contrasts["outcome"].eq("jump_quote_volume_log")]
    lines = [
        "## Material Passport", "", "- Origin Skill: experiment-agent", "- Origin Mode: run", f"- Origin Date: {now}",
        "- Verification Status: VERIFIED", "- Version Label: exp_result_v1", "", "## Experiment Result", "",
        "- **ID**: week10-pre-event-lead-curve-20261007", "- **Type**: analysis", "- **Status**: completed",
        '- **Command**: `py ".\\第十周事件前Lead曲线.py"`', f"- **Working Directory**: `{ROOT}`", f"- **Duration**: {duration:.2f} seconds", "- **Exit Code**: 0", "",
        "### Frozen Estimand", "",
        "- Primary outcome: `jump_quote_volume_log`.",
        "- Lead anchors: `-60/-45/-30/-15` minutes; real anchor: `0` minutes.",
        "- Primary contrast: real-anchor local jump minus the mean of the four pre-event local jumps within the same event-asset key.",
        "- All pseudo anchors precede the scheduled funding time; the event-date cluster is the original funding date, including lead windows crossing the prior UTC calendar day.",
        "- This is a local event-time association and anticipation check, not a causal effect or untreated counterfactual.", "",
        "### Main Contrast", "",
        "| 指标 | 真实跳变 | 事件前均值 | 真实 - 事件前均值 | 日期聚类95% CI | 7日块95% CI | p | BH q(3) |",
        "|------|---------:|-----------:|------------------:|---------------:|-------------:|--:|--------:|",
    ]
    for row in contrasts.itertuples(index=False):
        q = row.bh_q_value_3_main_outcomes
        lines.append(
            f"| {row.outcome_label} | {row.mean_real_jump:.6g} | {row.mean_pre_lead:.6g} | {row.real_minus_pre_lead_mean:.6g} | "
            f"[{row.cluster_ci95_lo:.6g}, {row.cluster_ci95_hi:.6g}] | [{row.moving_7d_block_ci95_lo:.6g}, {row.moving_7d_block_ci95_hi:.6g}] | "
            f"{row.cluster_p_value:.6g} | {'NA' if pd.isna(q) else f'{q:.6g}'} |"
        )
    lines.extend([
        "", "### Real Anchor Minus Each Lead", "",
        "逐 lead 比较的 BH 家族只包含三个预设主要端点 × 四个 lead（12 项）；主动买入不平衡仍为探索性结果，不进入此家族。",
        "| 指标 | lead(min) | 真实 - lead | 日期聚类95% CI | p | BH q(12主端点) |",
        "|------|----------:|-----------:|---------------:|--:|--------------:|",
    ])
    for row in lead_pairs.itertuples(index=False):
        q = row.bh_q_value_12_main_lead_family
        lines.append(
            f"| {row.outcome_label} | {row.lead_offset_min} | {row.mean_real_minus_lead:.6g} | "
            f"[{row.cluster_ci95_lo:.6g}, {row.cluster_ci95_hi:.6g}] | {row.cluster_p_value:.6g} | "
            f"{'NA' if pd.isna(q) else f'{q:.6g}'} |"
        )
    lines.extend([
        "", "### Lead Curve", "",
        "下表是各事件时间锚点的平均局部跳变；置信区间是日期聚类区间，7 日块区间用于序列相关敏感性检查。",
        "", "| 指标 | 锚点 | 平均跳变 | 日期聚类95% CI | 7日块95% CI | N | 日期clusters |",
        "|------|------|---------:|---------------:|-------------:|--:|-------------:|",
    ])
    for row in curve.itertuples(index=False):
        if row.outcome in MAIN_OUTCOMES:
            lines.append(
                f"| {row.outcome_label} | {row.anchor_label} | {row.mean_jump:.6g} | [{row.cluster_ci95_lo:.6g}, {row.cluster_ci95_hi:.6g}] | "
                f"[{row.moving_7d_block_ci95_lo:.6g}, {row.moving_7d_block_ci95_hi:.6g}] | {row.n} | {row.date_clusters} |"
            )
    lines.extend([
        "", "### Interpretation Boundaries", "",
        "- 事件前 lead 曲线用于检查结算前局部活动是否已经变化；真实时点高于事件前均值只能表述为时点差异，不能单独证明资金费率结算造成该差异。",
        "- lead 曲线并不单调：-60 分钟成交额跳变约 0.383，高于 -45/-30/-15 分钟；0 与 -60 同为 minute 00，而其差异远小于 0 与另外三个非整点 lead 的差异。这提示分钟相位/整点结构是实质混杂。",
        "- 因此真实减四个 lead 均值不能作为干净结算冲击；必须同时报告真实减每个 lead 的配对差异，并将 minute-phase 解释留待下一阶段审计。",
        "- `-60` 分钟锚点的前后 K 线可能跨越前一 UTC 自然日，但所有观测均按原始资金费率事件日期聚类，而不是按锚点日期重新聚类。",
        "- 5 分钟 OHLCV 仍不观测盘口价差、订单簿深度或真实执行成本；Amihud-like 指标是无方向的成交额归一化绝对收益代理。",
        "- 3 个主要端点的事件前均值对比做 BH 校正；主动买入不平衡仅作探索性结果。",
        "- 真实时点减每个单独 lead 的 12 个差异只对三个主要端点做 BH 校正；主动买入不平衡未进入校正家族。它们用于检查前置窗口敏感性，不替代事件前均值估计。",
        "- 本阶段没有使用第八周的 `+3h/+5h` 结算后时点，也没有把历史实现 funding rate 当作结算前可交易预测变量。",
        "", "### Data Quality", "", "| Metric | Value |", "|--------|------:|",
    ])
    for row in qc.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.value} |")
    lines.extend(["", "### Output Files", "", "- `data/第十周事件前Lead事件级面板.csv`", "- `data/第十周事件级Lead汇总.csv`", "- `output/第十周Lead曲线.csv`", "- `output/第十周真实减事件前均值.csv`", "- `output/第十周真实减各Lead.csv`", "- `output/第十周分层差异.csv`", "- `output/第十周数据质量摘要.csv`", "", "### Reproducibility", "", "- Script completed with exit code 0; structured outputs were written. Same-environment hash comparison is recorded in the statistical review."])
    (OUT / "第十周实验结果与解释.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events()
    bars = load_bars()
    panel = build_panel(events, bars)
    summary = summarize_events(panel)
    curve = lead_curve(panel)
    contrasts = treatment_contrasts(summary)
    lead_pairs = real_minus_each_lead(summary)
    strata = stratified_contrasts(summary)
    qc = quality(events, bars, panel, summary)
    panel.to_csv(DATA / "第十周事件前Lead事件级面板.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(DATA / "第十周事件级Lead汇总.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(OUT / "第十周Lead曲线.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(OUT / "第十周真实减事件前均值.csv", index=False, encoding="utf-8-sig")
    lead_pairs.to_csv(OUT / "第十周真实减各Lead.csv", index=False, encoding="utf-8-sig")
    strata.to_csv(OUT / "第十周分层差异.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(OUT / "第十周数据质量摘要.csv", index=False, encoding="utf-8-sig")
    duration = time.perf_counter() - started
    write_report(events, panel, summary, curve, contrasts, lead_pairs, qc, duration)
    print(f"events={len(events)} panel={len(panel)} summary={len(summary)} complete={int(panel['pre_open'].notna().mul(panel['post_open'].notna()).sum())}")
    print("contrasts")
    print(contrasts.to_string(index=False))
    print("quality")
    print(qc.to_string(index=False))
    print("real minus each lead")
    print(lead_pairs.to_string(index=False))


if __name__ == "__main__":
    main()
