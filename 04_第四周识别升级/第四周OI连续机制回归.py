# -*- coding: utf-8 -*-
"""Continuous BTC OI heterogeneity regression on paired event differences."""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"


def main() -> None:
    path = DATA / "第四周BTC事件级OI对齐.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["diff_car_0", "diff_car_p60", "pre_oi_value", "dOI_1h_pct", "abs_funding_bp", "event_date"]).copy()
    frame["log_oi_pre"] = np.log(frame["pre_oi_value"])
    frame["z_log_oi"] = (frame["log_oi_pre"] - frame["log_oi_pre"].mean()) / frame["log_oi_pre"].std(ddof=1)
    rows = []
    for outcome in ["diff_car_0", "diff_car_p60"]:
        model = smf.ols(f"{outcome} ~ z_log_oi + dOI_1h_pct + abs_funding_bp", data=frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame["event_date"]}
        )
        for term in ["Intercept", "z_log_oi", "dOI_1h_pct", "abs_funding_bp"]:
            rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "coefficient_bp": model.params.get(term),
                    "cluster_se_bp": model.bse.get(term),
                    "p_value": model.pvalues.get(term),
                    "ci95_lo_bp": model.conf_int().loc[term, 0],
                    "ci95_hi_bp": model.conf_int().loc[term, 1],
                    "n": int(model.nobs),
                    "date_clusters": int(frame["event_date"].nunique()),
                    "r_squared": model.rsquared,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第四周BTC_OI连续机制回归_日期聚类.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
