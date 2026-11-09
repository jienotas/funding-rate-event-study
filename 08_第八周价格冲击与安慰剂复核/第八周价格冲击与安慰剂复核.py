# -*- coding: utf-8 -*-
"""Eighth-week local price-impact and timing-placebo analysis."""

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
N_BOOT = 4000
SEED = 20261006
ANCHOR_OFFSETS = (0, 3, 5)

BAR_VALUE_COLUMNS = ("open", "high", "low", "close", "quote_volume", "taker_buy_quote")
OUTCOME_COLUMNS = {
    "jump_quote_volume_log": "log成交额局部变化",
    "jump_range_bp": "高低价振幅局部变化_bp",
    "jump_amihud_log1p": "log1p成交额归一化绝对收益代理局部变化",
    "jump_taker_imbalance": "主动买入不平衡局部变化",
}


def load_bars() -> pd.DataFrame:
    bars = pd.read_csv(BARS_PATH, usecols=["symbol", "timestamp", *BAR_VALUE_COLUMNS])
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    for column in BAR_VALUE_COLUMNS:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna(subset=["symbol", "timestamp"])
    if bars.duplicated(["symbol", "timestamp"]).any():
        duplicates = int(bars.duplicated(["symbol", "timestamp"], keep=False).sum())
        raise ValueError(f"K线存在重复 symbol-timestamp 键，共 {duplicates} 行，停止以避免静默错配。")
    return bars


def load_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_PATH)
    required = {"symbol", "timestamp", "group"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"事件索引缺少必要列：{sorted(missing)}")
    events = events.loc[events["group"].eq("A_real_settlement")].copy()
    events["funding_time"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events = events.dropna(subset=["symbol", "funding_time"])
    if events.duplicated(["symbol", "funding_time"]).any():
        raise ValueError("真实结算事件存在重复 symbol-timestamp 键，停止以避免重复计数。")
    events["funding_event_key"] = (
        events["symbol"].astype(str)
        + "|"
        + events["funding_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    events["event_date"] = events["funding_time"].dt.strftime("%Y-%m-%d")
    events["funding_hour_utc"] = events["funding_time"].dt.hour
    return events[["symbol", "funding_time", "funding_event_key", "event_date", "funding_hour_utc"]]


def construct_anchor_panel(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    anchor_frames = []
    for offset in ANCHOR_OFFSETS:
        frame = events.copy()
        frame["offset_hours"] = offset
        frame["anchor_label"] = "真实结算" if offset == 0 else f"结算后+{offset}h"
        frame["anchor_time"] = frame["funding_time"] + pd.to_timedelta(offset, unit="h")
        frame["pre_bar_time"] = frame["anchor_time"] - pd.Timedelta(minutes=5)
        frame["post_bar_time"] = frame["anchor_time"]
        anchor_frames.append(frame)
    anchors = pd.concat(anchor_frames, ignore_index=True)

    pre = bars.rename(columns={"timestamp": "pre_bar_time", **{c: f"pre_{c}" for c in BAR_VALUE_COLUMNS}})
    post = bars.rename(columns={"timestamp": "post_bar_time", **{c: f"post_{c}" for c in BAR_VALUE_COLUMNS}})
    panel = anchors.merge(pre, on=["symbol", "pre_bar_time"], how="left", validate="many_to_one")
    panel = panel.merge(post, on=["symbol", "post_bar_time"], how="left", validate="many_to_one")

    for side in ("pre", "post"):
        opening = panel[f"{side}_open"]
        closing = panel[f"{side}_close"]
        high = panel[f"{side}_high"]
        low = panel[f"{side}_low"]
        quote_volume = panel[f"{side}_quote_volume"]
        taker_buy = panel[f"{side}_taker_buy_quote"]
        valid_price = opening.gt(0) & closing.gt(0) & high.gt(0) & low.gt(0)
        valid_volume = quote_volume.gt(0)

        panel[f"{side}_abs_return_bp"] = np.where(
            valid_price, np.abs(np.log(closing / opening)) * 10000, np.nan
        )
        panel[f"{side}_range_bp"] = np.where(
            valid_price, np.log(high / low) * 10000, np.nan
        )
        panel[f"{side}_quote_volume_log"] = np.where(
            valid_volume, np.log(quote_volume), np.nan
        )
        panel[f"{side}_amihud_bp_per_million_usdt"] = np.where(
            valid_volume,
            panel[f"{side}_abs_return_bp"] / (quote_volume / 1_000_000),
            np.nan,
        )
        panel[f"{side}_amihud_log1p"] = np.log1p(
            panel[f"{side}_amihud_bp_per_million_usdt"].clip(lower=0)
        )
        panel[f"{side}_taker_imbalance"] = np.where(
            valid_volume, 2 * taker_buy / quote_volume - 1, np.nan
        )

    panel["jump_quote_volume_log"] = panel["post_quote_volume_log"] - panel["pre_quote_volume_log"]
    panel["jump_range_bp"] = panel["post_range_bp"] - panel["pre_range_bp"]
    panel["jump_amihud_log1p"] = panel["post_amihud_log1p"] - panel["pre_amihud_log1p"]
    panel["jump_amihud_raw"] = (
        panel["post_amihud_bp_per_million_usdt"]
        - panel["pre_amihud_bp_per_million_usdt"]
    )
    panel["jump_taker_imbalance"] = panel["post_taker_imbalance"] - panel["pre_taker_imbalance"]
    panel["anchor_hour_utc"] = panel["anchor_time"].dt.hour
    return panel


def paired_contrasts(panel: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["funding_event_key", "symbol", "event_date", "funding_hour_utc"]
    rows = []
    for outcome, label in OUTCOME_COLUMNS.items():
        wide = panel.pivot(index=index_columns, columns="offset_hours", values=outcome).reset_index()
        for offset in (3, 5):
            pair = wide[index_columns].copy()
            pair["outcome"] = outcome
            pair["outcome_label"] = label
            pair["placebo_offset_hours"] = offset
            pair["real_jump"] = wide[0]
            pair["placebo_jump"] = wide[offset]
            pair["real_minus_placebo"] = wide[0] - wide[offset]
            rows.append(pair)
    return pd.concat(rows, ignore_index=True)


def cluster_bootstrap(values: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    grouped = values.groupby("event_date")["real_minus_placebo"].agg(["sum", "count"])
    n_clusters = len(grouped)
    if n_clusters < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, n_clusters, size=(N_BOOT, n_clusters))
    sums = grouped["sum"].to_numpy()[draw_indices].sum(axis=1)
    counts = grouped["count"].to_numpy()[draw_indices].sum(axis=1)
    draws = sums / counts
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), float(draws.std(ddof=1))


def moving_block_bootstrap(
    values: pd.DataFrame, seed: int, block_days: int = 7
) -> tuple[float, float, float]:
    grouped = values.groupby("event_date")["real_minus_placebo"].agg(["sum", "count"])
    if grouped.empty:
        return np.nan, np.nan, np.nan
    first_day = pd.Timestamp(grouped.index.min())
    last_day = pd.Timestamp(grouped.index.max())
    calendar = pd.date_range(first_day, last_day, freq="D").strftime("%Y-%m-%d")
    grouped = grouped.reindex(calendar, fill_value=0)
    n_days = len(grouped)
    n_blocks = math.ceil(n_days / block_days)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_days, size=(N_BOOT, n_blocks))
    offsets = np.arange(block_days)
    sampled_days = (starts[:, :, None] + offsets[None, None, :]) % n_days
    sampled_days = sampled_days.reshape(N_BOOT, -1)[:, :n_days]
    sums = grouped["sum"].to_numpy()[sampled_days].sum(axis=1)
    counts = grouped["count"].to_numpy()[sampled_days].sum(axis=1)
    draws = sums[counts > 0] / counts[counts > 0]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), float(draws.std(ddof=1))


def estimate_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    estimates = []
    for outcome_index, outcome in enumerate(OUTCOME_COLUMNS):
        for offset in (3, 5):
            data = contrasts.loc[
                contrasts["outcome"].eq(outcome)
                & contrasts["placebo_offset_hours"].eq(offset)
            ].dropna(subset=["real_jump", "placebo_jump", "real_minus_placebo", "event_date"])
            if data.empty:
                continue
            model = sm.OLS(data["real_minus_placebo"], np.ones((len(data), 1))).fit(
                cov_type="cluster",
                cov_kwds={"groups": data["event_date"], "use_correction": True},
                use_t=True,
            )
            ci = np.asarray(model.conf_int(alpha=0.05))[0]
            bootstrap_lo, bootstrap_hi, bootstrap_se = cluster_bootstrap(
                data, SEED + 1000 * outcome_index + offset
            )
            block_lo, block_hi, block_se = moving_block_bootstrap(
                data, SEED + 2000 * outcome_index + offset
            )
            estimates.append(
                {
                    "outcome": outcome,
                    "outcome_label": OUTCOME_COLUMNS[outcome],
                    "placebo_offset_hours": offset,
                    "n_paired_event_asset": len(data),
                    "date_clusters": data["event_date"].nunique(),
                    "mean_real_jump": data["real_jump"].mean(),
                    "mean_placebo_jump": data["placebo_jump"].mean(),
                    "mean_real_minus_placebo": float(np.asarray(model.params)[0]),
                    "cluster_robust_se": float(np.asarray(model.bse)[0]),
                    "cluster_robust_ci95_lo": float(ci[0]),
                    "cluster_robust_ci95_hi": float(ci[1]),
                    "cluster_robust_p_value": float(np.asarray(model.pvalues)[0]),
                    "date_bootstrap_ci95_lo": bootstrap_lo,
                    "date_bootstrap_ci95_hi": bootstrap_hi,
                    "date_bootstrap_se": bootstrap_se,
                    "moving_7d_block_ci95_lo": block_lo,
                    "moving_7d_block_ci95_hi": block_hi,
                    "moving_7d_block_se": block_se,
                }
            )
    result = pd.DataFrame(estimates)
    result["fdr_bh_q_value_8_tests"] = np.nan
    valid = result["cluster_robust_p_value"].notna()
    if valid.any():
        result.loc[valid, "fdr_bh_q_value_8_tests"] = multipletests(
            result.loc[valid, "cluster_robust_p_value"], method="fdr_bh"
        )[1]
    return result


def build_stratified_estimates(contrasts: pd.DataFrame) -> pd.DataFrame:
    data = contrasts.dropna(subset=["real_minus_placebo"]).copy()
    rows = []
    for stratifier in ("symbol", "funding_hour_utc"):
        for (outcome, label, offset, stratum), part in data.groupby(
            ["outcome", "outcome_label", "placebo_offset_hours", stratifier],
            dropna=False,
        ):
            model = sm.OLS(part["real_minus_placebo"], np.ones((len(part), 1))).fit(
                cov_type="cluster",
                cov_kwds={"groups": part["event_date"], "use_correction": True},
                use_t=True,
            )
            ci = np.asarray(model.conf_int(alpha=0.05))[0]
            rows.append(
                {
                    "stratifier": stratifier,
                    "stratum": str(stratum),
                    "outcome": outcome,
                    "outcome_label": label,
                    "placebo_offset_hours": offset,
                    "n_paired_event_asset": len(part),
                    "date_clusters": part["event_date"].nunique(),
                    "mean_real_minus_placebo": float(np.asarray(model.params)[0]),
                    "cluster_robust_ci95_lo": float(ci[0]),
                    "cluster_robust_ci95_hi": float(ci[1]),
                }
            )
    return pd.DataFrame(rows)


def build_quality(events: pd.DataFrame, bars: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "real_funding_events", "value": len(events)},
        {"metric": "symbols", "value": ",".join(sorted(events["symbol"].astype(str).unique()))},
        {"metric": "raw_5m_bars", "value": len(bars)},
        {"metric": "raw_bar_time_min_utc", "value": bars["timestamp"].min().isoformat()},
        {"metric": "raw_bar_time_max_utc", "value": bars["timestamp"].max().isoformat()},
        {"metric": "anchor_rows_expected", "value": len(events) * len(ANCHOR_OFFSETS)},
        {"metric": "anchors_with_both_bars", "value": int(panel["pre_open"].notna().mul(panel["post_open"].notna()).sum())},
        {"metric": "anchors_missing_pre_bar", "value": int(panel["pre_open"].isna().sum())},
        {"metric": "anchors_missing_post_bar", "value": int(panel["post_open"].isna().sum())},
        {"metric": "zero_or_missing_pre_quote_volume", "value": int(panel["pre_quote_volume"].le(0).fillna(True).sum())},
        {"metric": "zero_or_missing_post_quote_volume", "value": int(panel["post_quote_volume"].le(0).fillna(True).sum())},
        {"metric": "date_clusters", "value": events["event_date"].nunique()},
    ]
    for offset in ANCHOR_OFFSETS:
        part = panel.loc[panel["offset_hours"].eq(offset)]
        rows.append(
            {
                "metric": f"offset_{offset}h_complete_pre_post",
                "value": int(part["pre_open"].notna().mul(part["post_open"].notna()).sum()),
            }
        )
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.6g}"


def write_report(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    results: pd.DataFrame,
    quality: pd.DataFrame,
    stratified: pd.DataFrame,
    duration_seconds: float,
) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        f"- Origin Date: {now}",
        "- Verification Status: UNVERIFIED",
        "- Version Label: exp_result_v1",
        "",
        "## Experiment Result",
        "",
        "- **ID**: week8-price-impact-placebo-20261006",
        "- **Type**: analysis",
        "- **Status**: completed",
        '- **Command**: `py ".\\第八周价格冲击与安慰剂复核.py"`',
        f"- **Working Directory**: `{ROOT}`",
        f"- **Duration**: {duration_seconds:.2f} seconds",
        "- **Exit Code**: 0",
        "",
        "### Output Files",
        "",
        "| File | Size |",
        "|------|------|",
        f"| `data/第八周事件级局部变化面板.csv` | {(DATA / '第八周事件级局部变化面板.csv').stat().st_size:,} bytes |",
        f"| `data/第八周真实事件与时点安慰剂配对差异.csv` | {(DATA / '第八周真实事件与时点安慰剂配对差异.csv').stat().st_size:,} bytes |",
        f"| `output/第八周日期聚类比较.csv` | {(OUT / '第八周日期聚类比较.csv').stat().st_size:,} bytes |",
        f"| `output/第八周分层日期聚类估计.csv` | {(OUT / '第八周分层日期聚类估计.csv').stat().st_size:,} bytes |",
        f"| `output/第八周数据质量摘要.csv` | {(OUT / '第八周数据质量摘要.csv').stat().st_size:,} bytes |",
        "| `output/第八周实验结果与解释.md` | report; size not embedded |",
        "",
        "### Output Summary",
        "",
        f"- 真实结算事件：{len(events):,} 个；资产：{', '.join(sorted(events['symbol'].unique()))}。",
        f"- 事件锚点面板：{len(panel):,} 行；结算日期 cluster：{events['event_date'].nunique():,} 个。",
        "- 主对比：真实结算锚点的局部 post-pre 变化，减去同一结算事件在 +3h 或 +5h 锚点的局部变化。",
        "- 日期聚类标准误、日期 cluster bootstrap 及 7 日循环移动块 bootstrap 区间均已计算；8 项对比统一做 Benjamini-Hochberg FDR 校正。",
        "- 分资产和 UTC 结算小时的分层日期聚类区间另存为探索性异质性检查，未作为额外主检验。",
        "",
        "| 指标 | 对比 | 差值 | 日期聚类 95% CI | 7日块 bootstrap 95% CI | p | BH q | N | 日期 clusters |",
        "|------|------|------:|----------------:|----------------------:|--:|-----:|--:|--------------:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.outcome_label} | 真实 - +{row.placebo_offset_hours}h | "
            f"{fmt(row.mean_real_minus_placebo)} | "
            f"[{fmt(row.cluster_robust_ci95_lo)}, {fmt(row.cluster_robust_ci95_hi)}] | "
            f"[{fmt(row.moving_7d_block_ci95_lo)}, {fmt(row.moving_7d_block_ci95_hi)}] | "
            f"{fmt(row.cluster_robust_p_value)} | {fmt(row.fdr_bh_q_value_8_tests)} | "
            f"{row.n_paired_event_asset} | {row.date_clusters} |"
        )
    lines.extend(
        [
            "",
            "### Heterogeneity by Asset and Funding Hour",
            "",
            "下表给出分资产、分真实结算 UTC 小时的真实锚点减安慰剂锚点差异及日期聚类区间。分层估计是探索性异质性检查，没有额外报告分层 p 值或进行分层 FDR 检验。",
            "",
            "| 分层 | 组别 | 指标 | 对比 | 差异 | 日期聚类 95% CI | N | clusters |",
            "|------|------|------|------|-----:|----------------:|--:|---------:|",
        ]
    )
    selected = stratified.loc[
        stratified["outcome"].isin(
            ["jump_quote_volume_log", "jump_range_bp", "jump_amihud_log1p"]
        )
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.stratifier} | {row.stratum} | {row.outcome_label} | 真实 - +{row.placebo_offset_hours}h | "
            f"{fmt(row.mean_real_minus_placebo)} | "
            f"[{fmt(row.cluster_robust_ci95_lo)}, {fmt(row.cluster_robust_ci95_hi)}] | "
            f"{row.n_paired_event_asset} | {row.date_clusters} |"
        )
    lines.extend(
        [
            "",
            "### Method and Interpretation",
            "",
            "- 每个锚点使用紧邻的前一根 `[t-5m,t)` 与后一根 `[t,t+5m)` 5 分钟 K 线；局部变化定义为 post bar 减 pre bar。",
            "- 成交额归一化绝对收益代理为 `abs(log(close/open)) * 10,000 / (quote_volume / 1,000,000)`，原始单位是每百万 USDT 成交额对应的绝对收益基点。推断使用该代理 `log1p` 值的前后差，再计算真实锚点减时点安慰剂锚点；估计量是变换后的 log-point 差异，不是价格基点或方向性买卖压力。",
            "- `+3h/+5h` 锚点位于真实结算之后，可能已受结算后动态影响；其结果只检查时点特异性，不代表无处理反事实，也不是随机对照。固定日内时段差异仍可能影响比较。",
            "- K 线 OHLCV 不含盘口价差、订单簿深度或真实执行成本。因此 Amihud-like 指标只能称为粗略的成交额归一化价格变动代理。",
            "- 所有结果均为观察性关联。不得据此声称资金费率结算导致价格冲击或流动性恶化。分层区间用于显示异质性，不另作分层 p 值或显著性结论。",
            "- BH 校正范围限于本周预先列出的 8 个结果×时点对比，不覆盖前七周探索过的其他端点或模型；整体研究仍属探索性。",
            "- 7 日块区间用于检查相邻日期序列相关敏感性；该区间采用循环移动块近似，不等同于完全指定的市场时间序列模型。",
            "",
            "### Data Quality",
            "",
            "| Metric | Value |",
            "|--------|------:|",
        ]
    )
    for row in quality.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.value} |")
    lines.extend(
        [
            "",
            "### Anomalies Detected",
            "",
            "- 分析仅在原始 5 分钟 K 线键唯一、结算时间戳可解析且资产价格/成交额有效时继续；缺失窗口和零成交记录已计入数据质量摘要。",
            "- 本运行报告不自行认证复现或识别有效性；复现核验与因果识别边界见配套统计解释审阅。",
            "",
        ]
    )
    (OUT / "第八周实验结果与解释.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events()
    bars = load_bars()
    panel = construct_anchor_panel(events, bars)
    contrasts = paired_contrasts(panel)
    results = estimate_contrasts(contrasts)
    quality = build_quality(events, bars, panel)
    stratified = build_stratified_estimates(contrasts)

    panel.to_csv(DATA / "第八周事件级局部变化面板.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(DATA / "第八周真实事件与时点安慰剂配对差异.csv", index=False, encoding="utf-8-sig")
    results.to_csv(OUT / "第八周日期聚类比较.csv", index=False, encoding="utf-8-sig")
    stratified.to_csv(OUT / "第八周分层日期聚类估计.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUT / "第八周数据质量摘要.csv", index=False, encoding="utf-8-sig")
    duration_seconds = time.perf_counter() - started
    write_report(events, panel, results, quality, stratified, duration_seconds)

    print(f"events={len(events)} anchors={len(panel)} complete={int(panel['pre_open'].notna().mul(panel['post_open'].notna()).sum())}")
    print(f"date_clusters={events['event_date'].nunique()} bootstrap_draws={N_BOOT}")
    print(results.to_string(index=False))
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
