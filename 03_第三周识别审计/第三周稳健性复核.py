# -*- coding: utf-8 -*-
"""Cluster-day bootstrap and local basis-jump sensitivity for week three."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WEEK1 = ROOT.parent / "01_第一周可行性测试"
WEEK2 = ROOT.parent / "02_第二周机制数据管道"
OUT = ROOT / "output"
SEED = 20261006
N_BOOT = 4000


def bootstrap_cluster_difference(real: pd.DataFrame, pseudo: pd.DataFrame, outcome: str) -> dict:
    """Resample UTC dates as clusters and preserve all events within a date."""
    real = real.dropna(subset=[outcome, "cluster_date"])
    pseudo = pseudo.dropna(subset=[outcome, "cluster_date"])
    dates = sorted(set(real["cluster_date"]) | set(pseudo["cluster_date"]))
    if len(dates) < 2:
        raise ValueError("not enough date clusters")
    observed = real[outcome].mean() - pseudo[outcome].mean()
    r_group = real.groupby("cluster_date")[outcome].agg(["sum", "count"]).reindex(dates, fill_value=0)
    p_group = pseudo.groupby("cluster_date")[outcome].agg(["sum", "count"]).reindex(dates, fill_value=0)
    # Avoid Python's process-randomized hash so the bootstrap is reproducible.
    outcome_seed = sum(ord(ch) for ch in outcome)
    rng = np.random.default_rng(SEED + outcome_seed)
    sampled = rng.integers(0, len(dates), size=(N_BOOT, len(dates)))
    r_mean = r_group["sum"].to_numpy()[sampled].sum(axis=1) / r_group["count"].to_numpy()[sampled].sum(axis=1)
    p_mean = p_group["sum"].to_numpy()[sampled].sum(axis=1) / p_group["count"].to_numpy()[sampled].sum(axis=1)
    draws = r_mean - p_mean
    se = float(draws.std(ddof=1))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "outcome": outcome,
        "difference_bp": float(observed),
        "cluster_bootstrap_se_bp": se,
        "lo95_bp": float(lo),
        "hi95_bp": float(hi),
        "mde80_bp": float(2.8 * se),
        "date_clusters": len(dates),
        "bootstrap_reps": N_BOOT,
        "equivalent_pm1bp": bool(lo > -1 and hi < 1),
    }


def price_cluster_bootstrap() -> pd.DataFrame:
    panel = pd.read_csv(WEEK1 / "output" / "事件面板.csv")
    events = pd.read_csv(WEEK1 / "output" / "事件索引.csv")
    paths = panel.pivot_table(index="event_id", columns="tau", values="cumulative_bp", aggfunc="first")
    paths = paths.rename(columns={-60: "car_m60", 0: "car_0", 60: "car_p60", 120: "car_p120"}).reset_index()
    level = events.merge(paths, on="event_id", how="inner")
    level["event_time"] = pd.to_datetime(level["timestamp"], utc=True, format="mixed")
    level["cluster_date"] = level["event_time"].dt.strftime("%Y-%m-%d")
    real = level.loc[level["group"] == "A_real_settlement"].copy()
    pseudo = level.loc[level["group"] == "B_pseudo_settlement"].copy()
    rows = [bootstrap_cluster_difference(real, pseudo, outcome) for outcome in ["car_m60", "car_0", "car_p60", "car_p120"]]
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第三周价格效应_日期聚类bootstrap.csv", index=False, encoding="utf-8-sig")
    return result


def basis_cluster_bootstrap() -> pd.DataFrame:
    path = WEEK2 / "output" / "第二周基差事件面板.csv"
    panel = pd.read_csv(path)
    panel["event_time"] = pd.to_datetime(panel["event_time"], utc=True, format="mixed")
    rows = []
    for metric in ["perp_spot_basis_bp", "perp_index_basis_bp"]:
        event = panel.pivot_table(index=["event_id", "symbol", "event_time"], columns="tau", values=metric, aggfunc="first").reset_index()
        event["cluster_date"] = event["event_time"].dt.strftime("%Y-%m-%d")
        for label, a, b in [("jump_m1_to_0", 0, -1), ("change_m10_to_0", 0, -10), ("change_m10_to_p60", 60, -10)]:
            values = event[[a, b, "cluster_date"]].dropna().copy()
            values["effect_bp"] = values[a] - values[b]
            dates = sorted(values["cluster_date"].unique())
            grouped = values.groupby("cluster_date")["effect_bp"].agg(["sum", "count"]).reindex(dates, fill_value=0)
            rng = np.random.default_rng(SEED + len(rows))
            sampled = rng.integers(0, len(dates), size=(N_BOOT, len(dates)))
            draws = grouped["sum"].to_numpy()[sampled].sum(axis=1) / grouped["count"].to_numpy()[sampled].sum(axis=1)
            se = float(draws.std(ddof=1))
            lo, hi = np.quantile(draws, [0.025, 0.975])
            rows.append({
                "metric": metric,
                "outcome": label,
                "mean_change_bp": float(values["effect_bp"].mean()),
                "cluster_bootstrap_se_bp": se,
                "lo95_bp": float(lo),
                "hi95_bp": float(hi),
                "mde80_bp": float(2.8 * se),
                "event_n": len(values),
                "date_clusters": len(dates),
                "bootstrap_reps": N_BOOT,
                "equivalent_pm1bp": bool(lo > -1 and hi < 1),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第三周基差效应_日期聚类bootstrap与局部跳跃.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    price = price_cluster_bootstrap()
    basis = basis_cluster_bootstrap()
    print("price cluster bootstrap")
    print(price.to_string(index=False))
    print("basis cluster bootstrap and local jump")
    print(basis.to_string(index=False))


if __name__ == "__main__":
    main()
