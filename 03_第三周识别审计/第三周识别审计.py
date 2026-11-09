# -*- coding: utf-8 -*-
"""Run third-week precision, equivalence, and proxy-measurement diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WEEK1 = ROOT.parent / "01_第一周可行性测试"
WEEK2 = ROOT.parent / "02_第二周机制数据管道"
DATA = ROOT / "data"
OUT = ROOT / "output"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ci_diff(real: pd.Series, pseudo: pd.Series) -> dict:
    real = pd.to_numeric(real, errors="coerce").dropna()
    pseudo = pd.to_numeric(pseudo, errors="coerce").dropna()
    diff = float(real.mean() - pseudo.mean())
    se = float(np.sqrt(real.var(ddof=1) / len(real) + pseudo.var(ddof=1) / len(pseudo)))
    mde80 = (1.96 + 0.84) * se
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    return {
        "real_mean_bp": real.mean(),
        "pseudo_mean_bp": pseudo.mean(),
        "difference_bp": diff,
        "lo95_bp": lo,
        "hi95_bp": hi,
        "se_bp": se,
        "mde80_bp": mde80,
        "real_n": len(real),
        "pseudo_n": len(pseudo),
        "equivalent_pm1bp": bool(lo > -1 and hi < 1),
        "equivalent_pm2bp": bool(lo > -2 and hi < 2),
    }


def price_precision() -> pd.DataFrame:
    panel = pd.read_csv(WEEK1 / "output" / "事件面板.csv")
    events = pd.read_csv(WEEK1 / "output" / "事件索引.csv")
    paths = panel.pivot_table(index="event_id", columns="tau", values="cumulative_bp", aggfunc="first")
    paths = paths.rename(columns={0: "car_0", 60: "car_p60", 120: "car_p120", -60: "car_m60"}).reset_index()
    level = events.merge(paths, on="event_id", how="inner")
    real = level.loc[level["group"] == "A_real_settlement"]
    pseudo = level.loc[level["group"] == "B_pseudo_settlement"]
    rows = []
    for outcome in ["car_m60", "car_0", "car_p60", "car_p120"]:
        result = ci_diff(real[outcome], pseudo[outcome])
        result["outcome"] = outcome
        rows.append(result)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第三周价格效应精度与等价性.csv", index=False, encoding="utf-8-sig")
    return result


def basis_precision() -> pd.DataFrame:
    path = WEEK2 / "output" / "第二周基差事件面板.csv"
    if not path.exists():
        return pd.DataFrame()
    panel = pd.read_csv(path)
    rows = []
    for metric in ["perp_spot_basis_bp", "perp_index_basis_bp"]:
        event_level = panel.pivot_table(index="event_id", columns="tau", values=metric, aggfunc="first")
        event_level["change_pre_to_0"] = event_level[0] - event_level[-10]
        event_level["change_pre_to_60"] = event_level[60] - event_level[-10]
        for outcome in ["change_pre_to_0", "change_pre_to_60"]:
            values = pd.to_numeric(event_level[outcome], errors="coerce").dropna()
            mean = float(values.mean())
            se = float(values.std(ddof=1) / np.sqrt(len(values)))
            rows.append(
                {
                    "metric": metric,
                    "outcome": outcome,
                    "mean_change_bp": mean,
                    "lo95_bp": mean - 1.96 * se,
                    "hi95_bp": mean + 1.96 * se,
                    "mde80_bp": (1.96 + 0.84) * se,
                    "n": len(values),
                    "equivalent_pm1bp": bool(mean - 1.96 * se > -1 and mean + 1.96 * se < 1),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第三周基差效应精度.csv", index=False, encoding="utf-8-sig")
    return result


def proxy_measurement() -> pd.DataFrame:
    proxy_path = DATA / "premium_proxy_1m.csv"
    if not proxy_path.exists():
        return pd.DataFrame()
    proxy = pd.read_csv(proxy_path, usecols=["timestamp", "symbol", "reconstructed_pre_funding_proxy", "premium_trailing_8h"])
    proxy["timestamp"] = pd.to_datetime(proxy["timestamp"], utc=True, format="mixed").dt.floor("min")
    proxy["reconstructed_pre_funding_proxy"] = pd.to_numeric(proxy["reconstructed_pre_funding_proxy"], errors="coerce")
    events = pd.read_csv(WEEK1 / "output" / "事件索引.csv")
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, format="mixed").dt.floor("min")
    events = events.loc[events["group"] == "A_real_settlement"].copy()
    events["proxy_time"] = events["timestamp"] - pd.Timedelta(minutes=1)
    merged = events.merge(proxy, left_on=["symbol", "proxy_time"], right_on=["symbol", "timestamp"], how="left")
    merged["funding_rate"] = pd.to_numeric(merged["funding_rate"], errors="coerce")
    merged["proxy_error_bp"] = 10000 * (merged["funding_rate"] - merged["reconstructed_pre_funding_proxy"])
    rows = []
    for symbol, part in merged.groupby("symbol"):
        valid = part.dropna(subset=["funding_rate", "reconstructed_pre_funding_proxy"])
        corr = valid["funding_rate"].corr(valid["reconstructed_pre_funding_proxy"]) if len(valid) > 2 else np.nan
        rows.append(
            {
                "symbol": symbol,
                "event_count": len(part),
                "proxy_coverage": len(valid) / len(part) if len(part) else np.nan,
                "correlation_realized_proxy": corr,
                "mae_bp": valid["proxy_error_bp"].abs().mean() * 1 if len(valid) else np.nan,
                "median_error_bp": valid["proxy_error_bp"].median() if len(valid) else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "第三周费率代理测量审计.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(DATA / "事件费率与premium_proxy对齐.csv", index=False, encoding="utf-8-sig")
    return result


def plot_mde(price: pd.DataFrame, basis: pd.DataFrame) -> None:
    if price.empty:
        return
    x = np.arange(len(price))
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    ax.errorbar(x, price["difference_bp"], yerr=1.96 * price["se_bp"], fmt="o-", color="#B42318", capsize=4)
    ax.axhspan(-1, 1, color="#12B76A", alpha=0.08, label="等价区间 ±1 bp")
    ax.axhline(0, color="#98A2B3", linewidth=0.8)
    ax.set_xticks(x, price["outcome"])
    ax.set_ylabel("真实事件 - 伪事件（基点）")
    ax.set_title("第三周：真实与伪事件差异及 ±1 bp 等价区间")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "第三周_效应精度与等价区间.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    price = price_precision()
    basis = basis_precision()
    proxy = proxy_measurement()
    plot_mde(price, basis)
    print("price diagnostics")
    print(price.to_string(index=False))
    print("basis diagnostics")
    print(basis.to_string(index=False))
    print("proxy diagnostics")
    print(proxy.to_string(index=False))


if __name__ == "__main__":
    main()
