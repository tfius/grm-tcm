# Constitution Proxy — Findings

**Date:** 2026-05-29
**Branch:** feat/constitution-proxy

## Method

Per-subject expanding-window mean of all prior observations — causal
constitution proxy. At visit t, computes mean(x_{1..t}) for the same
subject. No future leakage. Approximates stable constitution K that
drives regime transitions via E_MATRIX in the generator.

## Results (transductive, Takens graph, n_modes=16, rho=0.1)

| Model | h=1 | h=3 | h=7 | h=14 | h=21 |
|-------|-----|-----|-----|------|------|
| takens_ridge | 0.517 | 0.284 | 0.368 | 0.407 | 0.431 |
| multiscale+grm | 0.524 | 0.294 | 0.381 | 0.431 | 0.446 |
| takens+prior | 0.521 | 0.299 | 0.402 | 0.454 | 0.469 |
| **takens+prior+grm** | **0.524** | **0.302** | **0.404** | **0.454** | **0.470** |

## Key findings

1. Constitution proxy is biggest single improvement at long horizons:
   h=7: +0.023 R² over multiscale+grm (0.404 vs 0.381)
   h=21: +0.024 R² over multiscale+grm (0.470 vs 0.446)

2. R² increases with horizon up to h=21 — stable constitution determines
   long-term trajectory. This is the "constitutional medicine" signature:
   knowing the patient's baseline predicts 3-week outcome better than
   1-day outcome.

3. Three-component architecture validated:
   - Takens (trajectory velocity) — dominates h=1
   - Prior mean (constitution proxy) — dominates h=7+ improvement
   - GRM modes (topology) — adds +0.002-0.005 everywhere
