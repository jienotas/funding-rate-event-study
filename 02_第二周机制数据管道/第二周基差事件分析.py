# -*- coding: utf-8 -*-
"""Construct perp-spot/index basis and run second-week event diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WEEK1 = ROOT.parent / "01_第一周可行性测试"
DATA = ROOT / "data"
OUT = ROOT / "output"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_price(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed").dt.floor("min")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    return frame.dropna(subset=["timestamp", "symbol", "price"]).drop_duplicates(["timestamp", "symbol"])


def merge_basis() -> pd.DataFrame:
    perp = pd.read_csv(WEEK1 / "data" / "perp_1m.csv")
    perp["timestamp"] = pd.to_datetime(perp["timestamp"], utc=True, format="mixed").dt.floor("min")
    perp["price"] = pd.to_numeric(perp["price"], errors="coerce")
    perp = perp.rename(columns={"price": "perp_price"})
    pieces = [perp]
    for name, col in [("spot_1m.csv", "spot_price"), ("index_1m.csv", "index_price"), ("mark_1m.csv", "mark_price")]:
        pieces.append(load_price(name).rename(columns={"price": col}))
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on=["timestamp", "symbol"], how="left")
    for col in ["spot_price", "index_price", "mark_price"]:
        merged[f"{col}_ret_bp"] = 10000 * np.log(merged[col] / merged[col].groupby(merged["symbol"]).shift(1))
    merged["perp_spot_basis_bp"] = 10000 * np.log(merged["perp_price"] / merged["spot_price"])
    merged["perp_index_basis_bp"] = 10000 * np.log(merged["perp_price"] / merged["index_price"])
    merged["perp_mark_basis_bp"] = 10000 * np.log(merged["perp_price"] / merged["mark_price"])
    merged.to_csv(DATA / "perp_spot_index_basis_1m.csv", index=False, encoding="utf-8-sig")
    return merged


def event_paths(merged: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_csv(WEEK1 / "output" / "事件索引.csv")
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, format="mixed").dt.floor("min")
    real = events.loc[events["group"] == "A_real_settlement"].copy()
    rows = []
    tau_values = np.arange(-120, 121)
    for event in real.itertuples(index=False):
        part = merged.loc[merged["symbol"] == event.symbol].set_index("timestamp").sort_index()
        targets = event.timestamp + pd.to_timedelta(tau_values, unit="m")
        window = part.reindex(targets)
        if window["perp_spot_basis_bp"].isna().any() or window["perp_index_basis_bp"].isna().any():
            continue
        for tau, row in zip(tau_values, window.itertuples(index=False)):
            rows.append(
                {
                    "event_id": event.event_id,
                    "symbol": event.symbol,
                    "event_time": event.timestamp,
                    "tau": int(tau),
                    "perp_spot_basis_bp": float(row.perp_spot_basis_bp),
                    "perp_index_basis_bp": float(row.perp_index_basis_bp),
                    "perp_price": float(row.perp_price),
                    "spot_price": float(row.spot_price),
                    "index_price": float(row.index_price),
                }
            )
    panel = pd.DataFrame(rows)
    panel.to_csv(OUT / "第二周基差事件面板.csv", index=False, encoding="utf-8-sig")
    if panel.empty:
        return panel
    summary = panel.groupby("tau")[["perp_spot_basis_bp", "perp_index_basis_bp"]].agg(["mean", "std", "count"]).reset_index()
    for metric in ["perp_spot_basis_bp", "perp_index_basis_bp"]:
        mean = summary[(metric, "mean")]
        se = summary[(metric, "std")] / np.sqrt(summary[(metric, "count")])
        summary[(metric, "lo95")] = mean - 1.96 * se
        summary[(metric, "hi95")] = mean + 1.96 * se
    summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(OUT / "第二周基差事件路径.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=160, sharex=True)
    for ax, metric, title in zip(
        axes,
        ["perp_spot_basis_bp", "perp_index_basis_bp"],
        ["永续-现货基差", "永续-指数基差"],
    ):
        x = summary["tau"].to_numpy()
        y = summary[f"{metric}_mean"].to_numpy()
        lo = summary[f"{metric}_lo95"].to_numpy()
        hi = summary[f"{metric}_hi95"].to_numpy()
        ax.plot(x, y, color="#B42318", linewidth=2)
        ax.fill_between(x, lo, hi, color="#B42318", alpha=0.15)
        ax.axvline(0, color="#111827", linestyle="--", linewidth=1)
        ax.axhline(0, color="#98A2B3", linewidth=0.8)
        ax.set_ylabel("基点")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("相对结算时间 tau（分钟）")
    fig.suptitle("第二周：结算事件附近的永续-现货/指数基差")
    fig.tight_layout()
    fig.savefig(OUT / "第二周_基差事件路径.png", bbox_inches="tight")
    plt.close(fig)
    return panel


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    merged = merge_basis()
    panel = event_paths(merged)
    print("merged rows", len(merged), "valid basis event rows", len(panel))
    if not panel.empty:
        print("valid basis events", panel["event_id"].nunique())


if __name__ == "__main__":
    main()
