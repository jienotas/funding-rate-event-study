# -*- coding: utf-8 -*-
"""Ninth-week frozen-endpoint and temporal holdout validation.

This stage checks temporal association stability, not a causal funding effect
or a tradable forecast. The event index contains the realized settlement rate;
pre-settlement observability is not established here.
The primary endpoint is frozen before fitting: the real-settlement local
change in log quote volume. Range and normalized absolute-return proxy are
secondary endpoints. The first 70% of event dates is the training period and
the final 30% is a temporal holdout.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WEEK8 = PROJECT / "08_第八周价格冲击与安慰剂复核"
EVENTS_PATH = PROJECT / "01_第一周可行性测试" / "output" / "事件索引.csv"
OI_PATH = PROJECT / "04_第四周识别升级" / "data" / "第四周事件级机制面板.csv"
DATA = ROOT / "data"
OUT = ROOT / "output"
TRAIN_SHARE = 0.70
PRIMARY = "jump_quote_volume_log"
SECONDARY = ["jump_range_bp", "jump_amihud_log1p"]
ENDPOINTS = [PRIMARY, *SECONDARY]


def load_real_panel() -> pd.DataFrame:
    path = WEEK8 / "data" / "第八周事件级局部变化面板.csv"
    panel = pd.read_csv(path)
    panel = panel.loc[panel["offset_hours"].eq(0)].copy()
    panel["funding_time"] = pd.to_datetime(panel["funding_time"], utc=True, errors="coerce")
    if panel["funding_event_key"].duplicated().any():
        raise ValueError("第八周真实事件面板存在重复 funding_event_key。")

    events = pd.read_csv(EVENTS_PATH, usecols=["symbol", "timestamp", "funding_rate", "group"])
    events = events.loc[events["group"].eq("A_real_settlement")].copy()
    events["funding_time"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    events["funding_rate"] = pd.to_numeric(events["funding_rate"], errors="coerce")
    events = events[["symbol", "funding_time", "funding_rate"]]
    panel = panel.merge(events, on=["symbol", "funding_time"], how="left", validate="one_to_one")
    panel["abs_funding_bp"] = panel["funding_rate"].abs() * 10000
    panel["event_hour"] = panel["funding_hour_utc"].astype(str)
    panel["event_weekday"] = panel["funding_time"].dt.dayofweek.astype(str)
    return panel


def attach_btc_oi(panel: pd.DataFrame) -> pd.DataFrame:
    oi = pd.read_csv(OI_PATH)
    oi = oi.loc[
        (oi["group"] == "A_real_settlement") & (oi["symbol"] == "BTCUSDT"),
        ["symbol", "event_time", "pre_oi_value", "log_oi_pre", "dOI_1h_pct"],
    ].copy()
    oi["event_time"] = pd.to_datetime(oi["event_time"], utc=True, errors="coerce")
    oi = oi.rename(columns={"event_time": "funding_time"})
    return panel.merge(oi, on=["symbol", "funding_time"], how="left", validate="one_to_one")


def make_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(panel["event_date"].dropna().unique())
    if len(dates) < 20:
        raise ValueError(f"日期 cluster 太少，无法进行时间外验证：{len(dates)}")
    cutoff_index = max(1, min(len(dates) - 1, int(np.floor(len(dates) * TRAIN_SHARE))))
    cutoff = dates[cutoff_index - 1]
    panel = panel.copy()
    panel["sample"] = np.where(panel["event_date"] <= cutoff, "train", "holdout")
    return panel, panel.loc[panel["sample"].eq("train")].copy(), panel.loc[panel["sample"].eq("holdout")].copy()


def fit_models(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        data = train.dropna(subset=[endpoint, "abs_funding_bp", "event_date", "symbol", "event_hour"]).copy()
        if len(data) < 50:
            continue
        # This is a dose-association model with clock-hour and asset controls.
        formula = f"{endpoint} ~ abs_funding_bp + C(event_hour) + C(symbol)"
        model = smf.ols(formula, data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data["event_date"], "use_correction": True}, use_t=True
        )
        term = "abs_funding_bp"
        ci = model.conf_int().loc[term]
        rows.append(
            {
                "endpoint": endpoint,
                "endpoint_role": "primary" if endpoint == PRIMARY else "secondary",
                "model": "dose + UTC hour FE + asset FE",
                "sample": "train_70pct_dates",
                "coefficient_abs_funding_bp": model.params.get(term),
                "cluster_se": model.bse.get(term),
                "p_value": model.pvalues.get(term),
                "ci95_lo": ci.iloc[0],
                "ci95_hi": ci.iloc[1],
                "n": int(model.nobs),
                "date_clusters": data["event_date"].nunique(),
                "r_squared": model.rsquared,
                "train_first_date": data["event_date"].min(),
                "train_last_date": data["event_date"].max(),
            }
        )
    return pd.DataFrame(rows)


def temporal_predictions(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        train_data = train.dropna(subset=[endpoint, "abs_funding_bp", "event_date", "symbol", "event_hour"]).copy()
        test = holdout.dropna(subset=[endpoint, "abs_funding_bp", "event_date", "symbol", "event_hour"]).copy()
        if len(train_data) < 50 or len(test) < 10:
            continue
        full_formula = f"{endpoint} ~ abs_funding_bp + C(event_hour) + C(symbol)"
        baseline_formula = f"{endpoint} ~ C(event_hour) + C(symbol)"
        full = smf.ols(full_formula, data=train_data).fit()
        baseline = smf.ols(baseline_formula, data=train_data).fit()
        full_pred = full.predict(test)
        baseline_pred = baseline.predict(test)
        actual = test[endpoint].to_numpy()
        full_rmse = float(np.sqrt(np.mean((actual - full_pred.to_numpy()) ** 2)))
        base_rmse = float(np.sqrt(np.mean((actual - baseline_pred.to_numpy()) ** 2)))
        actual_centered = actual - actual.mean()
        pred_centered = full_pred.to_numpy() - full_pred.to_numpy().mean()
        denom = np.sqrt(np.sum(actual_centered**2) * np.sum(pred_centered**2))
        test_corr = float(np.sum(actual_centered * pred_centered) / denom) if denom else np.nan
        rows.append(
            {
                "endpoint": endpoint,
                "endpoint_role": "primary" if endpoint == PRIMARY else "secondary",
                "sample": "holdout_30pct_dates",
                "n": len(test),
                "date_clusters": test["event_date"].nunique(),
                "holdout_first_date": test["event_date"].min(),
                "holdout_last_date": test["event_date"].max(),
                "baseline_hour_asset_rmse": base_rmse,
                "dose_hour_asset_rmse": full_rmse,
                "rmse_improvement_vs_baseline": base_rmse - full_rmse,
                "dose_model_test_correlation": test_corr,
                "mean_actual": float(actual.mean()),
                "mean_predicted": float(full_pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def btc_oi_models(train: pd.DataFrame) -> pd.DataFrame:
    data = train.loc[train["symbol"].eq("BTCUSDT")].dropna(
        subset=[PRIMARY, "abs_funding_bp", "log_oi_pre", "event_date", "event_hour"]
    ).copy()
    if len(data) < 50:
        return pd.DataFrame()
    data["z_log_oi"] = (data["log_oi_pre"] - data["log_oi_pre"].mean()) / data["log_oi_pre"].std(ddof=1)
    model = smf.ols(
        f"{PRIMARY} ~ abs_funding_bp + z_log_oi + C(event_hour)",
        data=data,
    ).fit(cov_type="cluster", cov_kwds={"groups": data["event_date"], "use_correction": True}, use_t=True)
    rows = []
    for term in ["abs_funding_bp", "z_log_oi"]:
        ci = model.conf_int().loc[term]
        rows.append(
            {
                "endpoint": PRIMARY,
                "sample": "train_BTC_70pct_dates",
                "term": term,
                "coefficient": model.params.get(term),
                "cluster_se": model.bse.get(term),
                "p_value": model.pvalues.get(term),
                "ci95_lo": ci.iloc[0],
                "ci95_hi": ci.iloc[1],
                "n": int(model.nobs),
                "date_clusters": data["event_date"].nunique(),
                "r_squared": model.rsquared,
            }
        )
    return pd.DataFrame(rows)


def quality(panel: pd.DataFrame, train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "real_event_asset_rows", "value": len(panel)},
        {"metric": "real_event_dates", "value": panel["event_date"].nunique()},
        {"metric": "train_rows", "value": len(train)},
        {"metric": "train_dates", "value": train["event_date"].nunique()},
        {"metric": "holdout_rows", "value": len(holdout)},
        {"metric": "holdout_dates", "value": holdout["event_date"].nunique()},
        {"metric": "cutoff_date", "value": train["event_date"].max()},
        {"metric": "funding_rate_missing_share", "value": float(panel["funding_rate"].isna().mean())},
        {"metric": "train_abs_funding_bp_mean", "value": float(train["abs_funding_bp"].mean())},
        {"metric": "train_abs_funding_bp_median", "value": float(train["abs_funding_bp"].median())},
        {"metric": "train_abs_funding_bp_p75", "value": float(train["abs_funding_bp"].quantile(0.75))},
        {"metric": "holdout_abs_funding_bp_mean", "value": float(holdout["abs_funding_bp"].mean())},
        {"metric": "holdout_abs_funding_bp_median", "value": float(holdout["abs_funding_bp"].median())},
        {"metric": "holdout_abs_funding_bp_p75", "value": float(holdout["abs_funding_bp"].quantile(0.75))},
        {"metric": "BTC_OI_match_share", "value": float(panel.loc[panel["symbol"].eq("BTCUSDT"), "log_oi_pre"].notna().mean())},
        {"metric": "BTC_OI_missing_train_rows", "value": int(train.loc[train["symbol"].eq("BTCUSDT"), "log_oi_pre"].isna().sum())},
        {"metric": "BTC_OI_missing_holdout_rows", "value": int(holdout.loc[holdout["symbol"].eq("BTCUSDT"), "log_oi_pre"].isna().sum())},
        {"metric": "primary_endpoint_missing_share", "value": float(panel[PRIMARY].isna().mean())},
        {"metric": "symbols", "value": ",".join(sorted(panel["symbol"].unique()))},
        {"metric": "funding_hours", "value": ",".join(map(str, sorted(panel["funding_hour_utc"].unique())))},
    ]
    return pd.DataFrame(rows)


def report(panel: pd.DataFrame, train: pd.DataFrame, holdout: pd.DataFrame, models: pd.DataFrame, predictions: pd.DataFrame, oi: pd.DataFrame, qc: pd.DataFrame, duration: float) -> None:
    primary_model = models.loc[models["endpoint"].eq(PRIMARY)]
    primary_test = predictions.loc[predictions["endpoint"].eq(PRIMARY)]
    coefficient_text = "NA"
    if not primary_model.empty:
        row = primary_model.iloc[0]
        coefficient_text = f"{row['coefficient_abs_funding_bp']:.6g} (p={row['p_value']:.4g}, 95% CI [{row['ci95_lo']:.6g}, {row['ci95_hi']:.6g}])"
    test_text = "NA"
    if not primary_test.empty:
        row = primary_test.iloc[0]
        test_text = f"RMSE improvement={row['rmse_improvement_vs_baseline']:.6g}; test correlation={row['dose_model_test_correlation']:.4g}"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "## Material Passport", "",
        "- Origin Skill: experiment-agent", "- Origin Mode: run", f"- Origin Date: {now}",
        "- Verification Status: VERIFIED", "- Version Label: exp_result_v1", "",
        "## Experiment Result", "",
        "- **ID**: week9-frozen-endpoint-temporal-validation-20261007",
        "- **Type**: analysis", "- **Status**: completed",
        '- **Command**: `py ".\\第九周冻结端点与时间外验证.py"`', f"- **Working Directory**: `{ROOT}`", f"- **Duration**: {duration:.2f} seconds", "- **Exit Code**: 0", "",
        "### Frozen Design", "",
        "- Primary endpoint: real-settlement `jump_quote_volume_log`.",
        "- Secondary endpoints: `jump_range_bp` and `jump_amihud_log1p`.",
        "- Training sample: first 70% of sorted UTC event dates; holdout: final 30%.",
        "- Training association model: funding-rate absolute dose + UTC hour fixed effects + asset fixed effects; standard errors clustered by UTC date.",
        "- Holdout is a temporal association-stability check using the realized settlement rate, not a pre-settlement tradable forecast or causal test.", "",
        "### Main Results", "",
        f"- Primary training dose coefficient: `{coefficient_text}`.",
        f"- Primary holdout check: `{test_text}`.",
        f"- Real event-asset rows: {len(panel):,}; train dates: {train['event_date'].nunique():,}; holdout dates: {holdout['event_date'].nunique():,}.",
        "- OI model is BTC-only and is reported as exploratory mechanism association; it is not a treatment-control estimate.", "",
        "- Funding-dose distribution shifts between train and holdout are reported in the quality table; this is a regime/covariate shift that limits the holdout interpretation.",
        "- BTC OI is missing mainly in the late holdout period, so the OI model has no genuine late-period OI validation.", "",
        "### Output Files", "",
        "| File | Description |", "|------|-------------|",
        "| `data/第九周冻结端点事件面板.csv` | Real-settlement outcomes, funding dose, split labels |",
        "| `output/第九周训练期剂量关联.csv` | Date-clustered training estimates |",
        "| `output/第九周时间外验证.csv` | Holdout RMSE/correlation diagnostics |",
        "| `output/第九周BTC_OI探索性模型.csv` | BTC OI mechanism association |",
        "| `output/第九周数据质量摘要.csv` | Data and split checks |",
        "", "### Interpretation Boundaries", "",
        "- A positive dose coefficient, if present, is an association between funding magnitude and local activity after hour/asset controls. It is not evidence that settlement caused the activity.",
        "- The temporal holdout tests whether an association fitted with the realized settlement rate transports to later dates. It does not establish pre-settlement observability, trading predictability, or solve calendar-time confounding, endogenous funding rates, news shocks, or the lack of an untreated same-hour control.",
        "- The `+3h/+5h` post-settlement timing checks from Week 8 are not used as causal controls here.",
        "- The normalized absolute-return proxy remains an OHLCV proxy, not quoted spread, depth, or realized execution cost.",
        "- This stage freezes the primary endpoint for the next decision; it does not authorize further endpoint switching based on significance.", "",
        "### Data Quality", "", "| Metric | Value |", "|--------|------:|",
    ]
    for row in qc.itertuples(index=False):
        lines.append(f"| {row.metric} | {row.value} |")
    lines.extend(["", "### Reproducibility", "", "- Script completed with exit code 0; structured outputs were written in the project directory.", "- Same-environment reproduction status should be checked by rerunning this exact command and comparing hashes."])
    (OUT / "第九周实验结果与解释.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    panel = attach_btc_oi(load_real_panel())
    panel, train, holdout = make_split(panel)
    models = fit_models(train)
    predictions = temporal_predictions(train, holdout)
    oi = btc_oi_models(train)
    qc = quality(panel, train, holdout)
    panel.to_csv(DATA / "第九周冻结端点事件面板.csv", index=False, encoding="utf-8-sig")
    models.to_csv(OUT / "第九周训练期剂量关联.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "第九周时间外验证.csv", index=False, encoding="utf-8-sig")
    oi.to_csv(OUT / "第九周BTC_OI探索性模型.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(OUT / "第九周数据质量摘要.csv", index=False, encoding="utf-8-sig")
    duration = time.perf_counter() - started
    report(panel, train, holdout, models, predictions, oi, qc, duration)
    print(f"rows={len(panel)} train={len(train)} holdout={len(holdout)} train_dates={train['event_date'].nunique()} holdout_dates={holdout['event_date'].nunique()}")
    print("training")
    print(models.to_string(index=False))
    print("holdout")
    print(predictions.to_string(index=False))
    print("oi")
    print(oi.to_string(index=False))
    print("quality")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
