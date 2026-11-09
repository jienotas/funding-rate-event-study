# -*- coding: utf-8 -*-
"""Fifth-week paired basis mechanism and timestamp sensitivity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WEEK1 = PROJECT / "01_第一周可行性测试"
WEEK2 = PROJECT / "02_第二周机制数据管道"
WEEK4 = PROJECT / "04_第四周识别升级"
DATA = ROOT / "data"
OUT = ROOT / "output"
N_BOOT = 4000
SEED = 20261006


def load_event_level() -> pd.DataFrame:
    events = pd.read_csv(WEEK1 / "output" / "事件索引.csv")
    events["event_time_raw"] = pd.to_datetime(events["timestamp"], utc=True, format="mixed")
    events["event_time"] = events["event_time_raw"].dt.floor("min")
    panel = pd.read_csv(WEEK1 / "output" / "事件面板.csv", usecols=["event_id", "tau", "cumulative_bp"])
    paths = (
        panel.loc[panel["tau"].isin([-60, -10, -1, 0, 60, 120])]
        .pivot_table(index="event_id", columns="tau", values="cumulative_bp", aggfunc="first")
        .rename(columns={-60: "car_m60", -10: "car_m10", -1: "car_m1", 0: "car_0", 60: "car_p60", 120: "car_p120"})
        .reset_index()
    )
    level = events.merge(paths, on="event_id", how="inner")
    level["event_date"] = level["event_time"].dt.strftime("%Y-%m-%d")
    level["pair_id"] = np.where(level["group"].eq("B_pseudo_settlement"), level["event_id"] - 1080, level["event_id"])
    level["abs_funding_bp"] = pd.to_numeric(level["funding_rate"], errors="coerce").abs() * 10000
    return level


def load_basis() -> pd.DataFrame:
    path = WEEK2 / "data" / "perp_spot_index_basis_1m.csv"
    basis = pd.read_csv(path)
    basis["timestamp"] = pd.to_datetime(basis["timestamp"], utc=True, format="mixed").dt.floor("min")
    cols = ["timestamp", "symbol", "perp_spot_basis_bp", "perp_index_basis_bp"]
    return basis[cols].drop_duplicates(["timestamp", "symbol"])


def basis_event_panel(level: pd.DataFrame) -> pd.DataFrame:
    basis = load_basis()
    # Keep explicit per-symbol series to avoid silently mixing BTC and ETH.
    series = {}
    for symbol, part in basis.groupby("symbol"):
        indexed = part.set_index("timestamp").sort_index()
        for metric in ["perp_spot_basis_bp", "perp_index_basis_bp"]:
            series[(symbol, metric)] = indexed[metric]
    rows = []
    for event in level.itertuples(index=False):
        row = {"event_id": event.event_id, "pair_id": event.pair_id, "symbol": event.symbol, "group": event.group, "event_time": event.event_time, "event_date": event.event_date}
        valid = True
        for metric in ["perp_spot_basis_bp", "perp_index_basis_bp"]:
            s = series.get((event.symbol, metric))
            if s is None:
                valid = False
                continue
            for tau, name in [(-10, "pre"), (-1, "m1"), (0, "zero"), (60, "p60")]:
                value = s.get(event.event_time + pd.Timedelta(minutes=tau), np.nan)
                row[f"{metric}_{name}"] = value
            row[f"{metric}_jump_m1_to_0"] = row.get(f"{metric}_zero", np.nan) - row.get(f"{metric}_m1", np.nan)
            row[f"{metric}_change_m10_to_0"] = row.get(f"{metric}_zero", np.nan) - row.get(f"{metric}_pre", np.nan)
            row[f"{metric}_change_m10_to_p60"] = row.get(f"{metric}_p60", np.nan) - row.get(f"{metric}_pre", np.nan)
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "第五周事件级基差断点面板.csv", index=False, encoding="utf-8-sig")
    return result


def load_btc_oi() -> pd.DataFrame:
    path = WEEK4 / "data" / "第四周事件级机制面板.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["group"].eq("A_real_settlement")].copy()
    cols = ["event_id", "pair_id", "symbol", "group", "event_date", "pre_oi_value", "dOI_1h_pct", "log_oi_pre", "abs_funding_bp"]
    return frame.loc[frame["symbol"].eq("BTCUSDT"), [c for c in cols if c in frame.columns]].drop_duplicates("event_id")


def bootstrap_diff(frame: pd.DataFrame, outcome: str) -> dict:
    frame = frame.dropna(subset=[outcome, "event_date"]).copy()
    grouped = frame.groupby("event_date")[outcome].agg(["sum", "count"])
    dates = grouped.index.to_numpy()
    rng = np.random.default_rng(SEED + sum(map(ord, outcome)))
    sampled = rng.integers(0, len(dates), size=(N_BOOT, len(dates)))
    draws = grouped["sum"].to_numpy()[sampled].sum(axis=1) / grouped["count"].to_numpy()[sampled].sum(axis=1)
    return {
        "outcome": outcome,
        "n": len(frame),
        "date_clusters": len(dates),
        "mean_bp": float(frame[outcome].mean()),
        "lo95_bp": float(np.quantile(draws, 0.025)),
        "hi95_bp": float(np.quantile(draws, 0.975)),
        "mde80_bp": float(2.8 * draws.std(ddof=1)),
    }


def fit_basis_models(level: pd.DataFrame, basis_panel: pd.DataFrame) -> pd.DataFrame:
    real = level.loc[level["group"].eq("A_real_settlement")].merge(basis_panel, on=["event_id", "pair_id", "symbol", "group", "event_time", "event_date"], how="inner")
    pseudo = level.loc[level["group"].eq("B_pseudo_settlement")].merge(basis_panel, on=["event_id", "pair_id", "symbol", "group", "event_time", "event_date"], how="inner")
    real = real.merge(load_btc_oi().loc[:, ["pair_id", "pre_oi_value", "dOI_1h_pct", "log_oi_pre", "abs_funding_bp"]], on="pair_id", how="left", suffixes=("", "_oi"))
    rows = []
    for metric in ["perp_spot_basis_bp", "perp_index_basis_bp"]:
        for outcome in [f"{metric}_jump_m1_to_0", f"{metric}_change_m10_to_0", f"{metric}_change_m10_to_p60"]:
            r = real[["pair_id", "event_date", outcome, "pre_oi_value", "dOI_1h_pct", "log_oi_pre", "abs_funding_bp"]].rename(columns={outcome: "real_value"})
            p = pseudo[["pair_id", outcome]].rename(columns={outcome: "pseudo_value"})
            pair = r.merge(p, on="pair_id", how="inner")
            pair["diff_basis_bp"] = pair["real_value"] - pair["pseudo_value"]
            pair["z_log_oi"] = (pair["log_oi_pre"] - pair["log_oi_pre"].mean()) / pair["log_oi_pre"].std(ddof=1)
            pair = pair.dropna(subset=["diff_basis_bp", "z_log_oi", "dOI_1h_pct", "abs_funding_bp"])
            if len(pair) < 30:
                continue
            summary = bootstrap_diff(pair, "diff_basis_bp")
            summary.update({"metric": metric, "model_outcome": outcome})
            rows.append(summary)
            model = smf.ols("diff_basis_bp ~ z_log_oi + dOI_1h_pct + abs_funding_bp", data=pair).fit(cov_type="cluster", cov_kwds={"groups": pair["event_date"]})
            for term in ["z_log_oi", "dOI_1h_pct", "abs_funding_bp"]:
                rows.append({"metric": metric, "model_outcome": outcome, "outcome": "regression", "term": term, "coefficient_bp": model.params.get(term), "cluster_se_bp": model.bse.get(term), "p_value": model.pvalues.get(term), "ci95_lo_bp": model.conf_int().loc[term, 0], "ci95_hi_bp": model.conf_int().loc[term, 1], "n": int(model.nobs), "date_clusters": int(pair["event_date"].nunique()), "r_squared": model.rsquared})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第五周基差断点_OI机制模型.csv", index=False, encoding="utf-8-sig")
    return result


def price_timing_sensitivity(level: pd.DataFrame) -> pd.DataFrame:
    prices = pd.read_csv(WEEK1 / "data" / "perp_1m.csv")
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True, format="mixed").dt.floor("min")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    series = {symbol: part.drop_duplicates("timestamp").set_index("timestamp")["price"].sort_index() for symbol, part in prices.groupby("symbol")}
    rows = []
    for event in level.itertuples(index=False):
        s = series.get(event.symbol)
        if s is None:
            continue
        for shift in [-1, 0, 1]:
            anchor = event.event_time + pd.Timedelta(minutes=shift)
            base = s.reindex(anchor + pd.to_timedelta(np.arange(-10, 0), unit="m"))
            targets = s.reindex(anchor + pd.to_timedelta([0, 60], unit="m"))
            if base.isna().any() or targets.isna().any():
                continue
            baseline = float(base.median())
            rows.append({"event_id": event.event_id, "pair_id": event.pair_id, "symbol": event.symbol, "group": event.group, "event_date": event.event_date, "shift_minutes": shift, "car_0": 10000 * np.log(targets.iloc[0] / baseline), "car_p60": 10000 * np.log(targets.iloc[1] / baseline)})
    raw = pd.DataFrame(rows)
    raw.to_csv(DATA / "第五周价格时间定位事件级.csv", index=False, encoding="utf-8-sig")
    real = raw.loc[raw["group"].eq("A_real_settlement")]
    pseudo = raw.loc[raw["group"].eq("B_pseudo_settlement")]
    result_rows = []
    for shift in [-1, 0, 1]:
        r = real.loc[real["shift_minutes"].eq(shift), ["pair_id", "event_date", "car_0", "car_p60"]]
        p = pseudo.loc[pseudo["shift_minutes"].eq(shift), ["pair_id", "car_0", "car_p60"]].rename(columns={"car_0": "p_car_0", "car_p60": "p_car_p60"})
        pair = r.merge(p, on="pair_id", how="inner")
        pair["diff_car_0"] = pair["car_0"] - pair["p_car_0"]
        pair["diff_car_p60"] = pair["car_p60"] - pair["p_car_p60"]
        for outcome in ["diff_car_0", "diff_car_p60"]:
            row = bootstrap_diff(pair, outcome)
            row["shift_minutes"] = shift
            result_rows.append(row)
    result = pd.DataFrame(result_rows)
    result.to_csv(OUT / "第五周价格时间定位敏感性.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    level = load_event_level()
    basis_panel = basis_event_panel(level)
    basis_result = fit_basis_models(level, basis_panel)
    timing = price_timing_sensitivity(level)
    print("basis mechanism")
    print(basis_result.to_string(index=False) if not basis_result.empty else "empty")
    print("price timing sensitivity")
    print(timing.to_string(index=False) if not timing.empty else "empty")


if __name__ == "__main__":
    main()
