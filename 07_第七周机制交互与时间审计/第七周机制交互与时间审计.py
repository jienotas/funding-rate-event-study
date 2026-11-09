# -*- coding: utf-8 -*-
"""Seventh-week interaction and schedule-structure audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WEEK4 = PROJECT / "04_第四周识别升级"
WEEK6 = PROJECT / "06_第六周流动性机制"
DATA = ROOT / "data"
OUT = ROOT / "output"
N_BOOT = 4000
SEED = 20261006


def load_panel() -> pd.DataFrame:
    path = WEEK6 / "data" / "第六周配对流动性机制面板.csv"
    panel = pd.read_csv(path)
    oi = pd.read_csv(WEEK4 / "data" / "第四周事件级机制面板.csv")
    oi = oi.loc[(oi["group"] == "A_real_settlement") & (oi["symbol"] == "BTCUSDT"), ["pair_id", "pre_oi_value", "log_oi_pre", "dOI_1h_pct"]]
    panel = panel.merge(oi, on="pair_id", how="left")
    panel["event_hour_utc"] = panel["event_hour_utc"].astype(str)
    panel["z_log_oi"] = (panel["log_oi_pre"] - panel["log_oi_pre"].mean()) / panel["log_oi_pre"].std(ddof=1)
    panel["z_dOI_1h"] = (panel["dOI_1h_pct"] - panel["dOI_1h_pct"].mean()) / panel["dOI_1h_pct"].std(ddof=1)
    panel["z_abs_funding"] = (panel["abs_funding_bp"] - panel["abs_funding_bp"].mean()) / panel["abs_funding_bp"].std(ddof=1)
    panel["oi_funding_interaction"] = panel["z_log_oi"] * panel["z_abs_funding"]
    return panel


def bootstrap_mean(frame: pd.DataFrame, outcome: str) -> dict:
    frame = frame.dropna(subset=[outcome, "event_date"])
    grouped = frame.groupby("event_date")[outcome].agg(["sum", "count"])
    dates = grouped.index.to_numpy()
    rng = np.random.default_rng(SEED + sum(map(ord, outcome)))
    sampled = rng.integers(0, len(dates), size=(N_BOOT, len(dates)))
    draws = grouped["sum"].to_numpy()[sampled].sum(axis=1) / grouped["count"].to_numpy()[sampled].sum(axis=1)
    return {"n": len(frame), "date_clusters": len(dates), "mean_bp_or_unit": float(frame[outcome].mean()), "lo95": float(np.quantile(draws, 0.025)), "hi95": float(np.quantile(draws, 0.975)), "mde80": float(2.8 * draws.std(ddof=1))}


def interaction_models(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = ["diff_post_quote_volume_log", "diff_post_taker_imbalance", "diff_post_range_bp", "diff_car_0", "diff_car_p60"]
    terms = ["z_log_oi", "z_dOI_1h", "z_abs_funding", "oi_funding_interaction"]
    for outcome in outcomes:
        data = panel.dropna(subset=[outcome, "z_log_oi", "z_dOI_1h", "z_abs_funding", "event_date", "symbol", "event_hour_utc"]).copy()
        if len(data) < 50:
            continue
        model = smf.ols(
            f"{outcome} ~ z_log_oi + z_dOI_1h + z_abs_funding + oi_funding_interaction + C(symbol) + C(event_hour_utc)",
            data=data,
        ).fit(cov_type="cluster", cov_kwds={"groups": data["event_date"]})
        for term in terms:
            rows.append({"outcome": outcome, "term": term, "coefficient": model.params.get(term), "cluster_se": model.bse.get(term), "p_value": model.pvalues.get(term), "ci95_lo": model.conf_int().loc[term, 0], "ci95_hi": model.conf_int().loc[term, 1], "n": int(model.nobs), "date_clusters": int(data["event_date"].nunique()), "r_squared": model.rsquared})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第七周机制交互回归_日期聚类.csv", index=False, encoding="utf-8-sig")
    return result


def hour_structure(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = ["diff_post_quote_volume_log", "diff_post_taker_imbalance", "diff_post_range_bp", "diff_car_0", "diff_car_p60"]
    for (symbol, hour), part in panel.groupby(["symbol", "event_hour_utc"], dropna=False):
        for outcome in outcomes:
            summary = bootstrap_mean(part, outcome)
            summary.update({"symbol": symbol, "event_hour_utc": hour, "outcome": outcome})
            rows.append(summary)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第七周真实事件小时结构_日期聚类.csv", index=False, encoding="utf-8-sig")
    return result


def save_quality(panel: pd.DataFrame) -> None:
    rows = [
        {"metric": "paired_rows", "value": len(panel)},
        {"metric": "date_clusters", "value": panel["event_date"].nunique()},
        {"metric": "BTC_OI_match_share", "value": panel["pre_oi_value"].notna().mean()},
        {"metric": "symbols", "value": panel["symbol"].nunique()},
        {"metric": "event_hours", "value": ",".join(sorted(panel["event_hour_utc"].dropna().unique()))},
    ]
    pd.DataFrame(rows).to_csv(OUT / "第七周数据质量摘要.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    panel.to_csv(DATA / "第七周机制交互面板.csv", index=False, encoding="utf-8-sig")
    save_quality(panel)
    interaction = interaction_models(panel)
    hours = hour_structure(panel)
    print("panel", len(panel), "OI matched", int(panel["pre_oi_value"].notna().sum()))
    print("interaction")
    print(interaction.to_string(index=False))
    print("hours")
    print(hours.to_string(index=False))


if __name__ == "__main__":
    main()
