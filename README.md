# Funding-Rate Settlement and Perpetual Price Pressure

An event-time study of what actually happens around crypto perpetual-futures funding
settlements — and a documented failure to identify a causal price-pressure effect.

**Status:** closed. Descriptive event-time results hold; the causal design does not.

---

## The question

BTC/ETH perpetuals settle funding every 8 hours (00:00 / 08:00 / 16:00 UTC).
The folk claim in the market is that settlement forces leveraged positions to close,
which generates measurable price pressure and a liquidity dip.

If that is true, it should be visible as a discontinuity around the settlement minute,
and it should be identifiable — not merely correlated with the time of day.

This repository is the full audit trail of trying to establish that, week by week,
including the parts that killed the design.

## Headline results

**1. Activity and volatility do rise after settlement — and survive placebo.**

1,080 BTC/ETH settlement events across 181 event dates. Comparing the local
post-minus-pre change at the true settlement anchor against the same event at
a placebo anchor 3 hours later:

| Outcome | True − placebo(+3h) | 95% CI (date-clustered) | BH q |
|---|---:|---|---:|
| log turnover | **+0.156** | [0.087, 0.226] | 0.0001 |
| high-low range (bp) | **+1.57** | [0.40, 2.74] | 0.023 |
| turnover-normalized \|return\| proxy | −0.011 | [−0.042, 0.020] | 0.48 |

The same contrasts against a +5h placebo are weaker and not significant for range
(0.73 bp, q = 0.37), which is the first hint that "3 hours later" is not a clean
counterfactual — the effect decays within a few hours.

**2. Price direction does not respond.** The normalized absolute-return proxy is
flat in every specification. Volume and range move; the price path does not.

**3. The causal claim fails on identification, not on statistics.**

A settlement event study needs variation in *when* settlement happens, or a clean
control group. Neither exists:

- Binance and Bybit BTC/ETH both settle every 8 hours, aligned. All 543 adjacent
  intervals per asset-exchange are exactly 8h; 544/544 events pair minute-to-minute
  across the two venues.
- OKX exposes only ~3 months of history through its public endpoint (300 records
  retrieved, earliest 2026-06-29 08:00 UTC). Inside that window all 299 adjacent
  intervals are also 8h.
- Cross-venue timestamps match to within 26 ms. There is no staggered settlement,
  no interval change, no usable treatment variation.

**4. The "effect" is partly a clock effect.** Minute-of-hour activity is periodic:
ordinary round hours show their own turnover bump. The UTC 00:00 slot is confounded
with the general hour boundary, so settlement cannot be separated from the clock
without an instrument that this data does not contain.

Correction, stated plainly: without an official, non-market-state-driven rule change
and a clean unaffected control contract, there is no defensible causal design here.
Adding order-book depth or a fancier model does not fix missing treatment variation.

## Why publish a negative result

Because the reusable part is the audit, not the finding. What this repo demonstrates:

- pre-specified placebo anchors and Benjamini-Hochberg correction across 8 contrasts,
  rather than picking the specification that looked best;
- date-clustered standard errors **and** 7-day circular block bootstrap, reported side
  by side;
- equivalence/precision bounds instead of "not significant";
- an explicit identification gate (week 12–14) that a design must pass before any
  causal language is allowed;
- a written decision to stop, with the reason, instead of escalating to weaker claims.

## Layout

| Folder | Content |
|---|---|
| `01_第一周可行性测试` | 1-minute perpetual/funding download, three-way event plots, first diagnostics |
| `02_第二周机制数据管道` | spot / index / mark prices, OKX predicted-funding archiver, basis event analysis |
| `03_第三周识别审计` | premium-proxy reconstruction, effect precision and equivalence bounds, date-cluster bootstrap |
| `04_第四周识别升级` | OKX field-semantics audit, open-interest mechanism regression, same-day time placebo |
| `05_第五周分钟定位敏感性` | sensitivity of the effect to minute-level event anchoring |
| `05_第五周正式机制模型` | formal mechanism model with an open-interest break |
| `06_第六周流动性机制` | 5-minute futures liquidity download and mechanism panel |
| `07_第七周机制交互与时间审计` | mechanism interactions, hour-of-day structure |
| `08_第八周价格冲击与安慰剂复核` | main price-impact vs placebo estimation, FDR, block bootstrap |
| `09_第九周冻结端点与时间外验证` | frozen-endpoint specification and out-of-sample check |
| `10_第十周事件前Lead曲线` | pre-event lead curves (effect already visible at −60m) |
| `11_第十一周时钟相位与同槽位审计` | minute-phase and same-slot clock audit |
| `12_第十二周数据可得性与识别闸门` | data-availability audit, identification gate |
| `13_第十三周跨所结算时点识别审计` | cross-exchange settlement-timestamp audit (PowerShell) |
| `14_第十四周历史结算规则审计` | historical funding-interval audit over 2026-04 to 2026-10 |
| `docs/research_design.md` | original research design and literature review |
| `docs/14week_review.pdf` | 14-week retrospective report |

Each week folder is self-contained: `data/` for inputs, `output/` for results,
scripts at the folder root.

## Data

Raw market data is **not** redistributed here (multi-GB of 1-minute klines).
Download scripts are included: `01_.../下载_binance_可行性数据.py`,
`02_.../下载现货指数与规则.py`, `06_.../第六周流动性下载与机制分析.py`.
Everything is public Binance/OKX endpoints; no API key required for the historical
endpoints used.

`output/` contains the result tables and figures the analysis produced, so the
numbers above can be checked without re-running the downloads. Two intermediate
panels (`事件面板.csv`, `第二周基差事件面板.csv`) exceed 5 MB and are excluded.

One external input comes from a separate project and is not public:
`01_.../第一周诊断分析.py` reads an open-interest metrics file through the
`METRICS_CSV` environment variable, falling back to `data/binance_metrics_5m_BTCUSDT.csv.gz`.

## Reproducing

```bash
pip install -r requirements.txt
cd 01_第一周可行性测试 && python 下载_binance_可行性数据.py
cd ../02_第二周机制数据管道 && python 下载现货指数与规则.py
cd ../01_第一周可行性测试 && python 三组事件图_最小实现.py && python 第一周诊断分析.py
# subsequent weeks read the previous weeks' output/ folders
```

Weeks 13 and 14 are PowerShell 7 scripts under `scripts/`; they call the official
exchange endpoints and paginate the history.

## Conventions used throughout

- Event minute = settlement timestamp; window = local pre/post change, not raw level.
- Placebo = the same event re-anchored at +3h or +5h, which absorbs the time-of-day
  and day-level shocks that a plain pre/post comparison would attribute to settlement.
- Inference = standard errors clustered on settlement date; 7-day circular moving-block
  bootstrap as a dependence-robust cross-check; BH-FDR within each family of contrasts.
- Every week writes a data-quality summary alongside its estimates.

## Limitations

- OHLCV-derived liquidity proxies (turnover, trade count, aggressive-buy proxy) are not
  quoted depth, spread, or execution cost.
- Spot data on the venues used carries price only, no volume or trade direction.
- The observation window is 2026-04 to 2026-10; earlier OKX history is not observable
  through public endpoints.

## License

MIT — see `LICENSE`.
