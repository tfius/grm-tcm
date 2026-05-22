# PCA Baseline Integration — Findings

**Date:** 2026-05-22
**Branch:** feat/pca-baseline
**Motivation:** Pang et al. (2023, Nature) benchmarked geometric eigenmodes against PCA of functional data (Extended Data Fig. 4). Same scrutiny must apply to GRM-TCM: prove that graph topology adds value beyond linear variance extraction.

## Hypotheses

- **H₀:** PCA(X_raw) → Ridge/Logistic matches GRM embeddings → Ridge/Logistic.
- **H₁:** GRM significantly outperforms PCA, especially on aliased-pair subset where trajectory/topology should matter.

## Method

- `PCA(n_components=n_modes)` fit on train split only, applied to full observation matrix.
- Same downstream heads (Ridge, Logistic) and lag augmentation as GRM.
- Evaluated on: regression, classification, flare onset, aliased-pair subset.
- Both transductive and inductive paths updated.

## Results (transductive diagnostic)

| Task | GRM+lag | PCA+lag | Winner |
|------|---------|---------|--------|
| Regression R² | 0.5643 | **0.5789** | PCA |
| Classification AUC | 0.6543 | 0.6536 | ~tie |
| Onset AUC (hard) | 0.6211 | 0.6195 | ~tie |
| Aliased R² | 0.4446 | **0.4510** | PCA |
| Aliased AUC | **0.6636** | 0.6621 | ~tie |

Without lag features:

| Task | GRM | PCA | Winner |
|------|-----|-----|--------|
| Regression R² | 0.2379 | **0.3635** | PCA by +0.13 |
| Classification AUC | 0.6210 | 0.6239 | ~tie |

## Interpretation

**H₀ holds.** PCA matches or slightly beats GRM across all tasks, including the aliased-pair subset where GRM's graph topology should theoretically provide an advantage.

Key observations:

1. **Lag features dominate.** Both GRM and PCA jump from ~0.3 to ~0.57 R² when yesterday's score is appended. The persistence signal is doing the heavy lifting, not the embedding.

2. **Without lag, PCA beats GRM.** PCA R²=0.36 vs GRM R²=0.24. The graph Laplacian eigenfilter is actually *losing* information compared to a simple linear projection. This suggests the GRM spectral weighting `g_k = 1/(1+ρ²λ_k)` is over-smoothing useful high-frequency variance.

3. **Aliased subset shows no GRM advantage.** This is the most damning result. The aliased pairs (stressed_recoverable vs stuck_agitated) are designed so that today's observations alias but futures diverge. GRM should separate them via graph position. It doesn't.

4. **Persistence baseline remains the hidden winner.** R²=0.4776 without any embedding, just "yesterday's score predicts today's score." Biological inertia dominates next-day prediction.

## What this means for the project

The graph Laplacian is currently acting as an expensive linear variance filter. The temporal and treatment edges in the graph are not contributing detectable predictive signal through the current Ridge/Logistic heads.

Possible explanations:
- Linear heads can't exploit nonlinear manifold structure (GRM topology exists but Ridge can't see it)
- The spectral filter ρ over-smooths, destroying discriminative high-frequency signal
- Temporal edge weight (0.75) is not strong enough to encode trajectory direction
- The graph construction hyperparameters are not tuned (no ablation study)

## Recommended next steps (in priority order)

1. **Delay embedding:** Concatenate `[x_t, x_{t-1}, x_{t-2}]` before projection. Tests whether trajectory information helps at all — cheapest possible test of temporal hypothesis.
2. **Smoothed-delta target:** Predict `MA(y,3)_{t+1} - MA(y,3)_t` instead of `y_{t+1}`. Removes persistence dominance, forces model to predict change.
3. **Horizon sweep:** Evaluate at horizons [1, 3, 7, 14] days. Persistence degrades at longer horizons; GRM might win at h≥3.
4. **Nonlinear heads:** Try gradient-boosted trees on GRM embeddings. If GRM+XGBoost >> PCA+XGBoost, the graph structure exists but linear heads can't access it.
5. **ρ sweep and edge ablation:** Tune the spectral filter and measure contribution of each edge type.

## References

- Pang, J.C. et al. (2023). Geometric constraints on human brain function. *Nature*, 618, 566–574. Extended Data Fig. 4: PCA vs geometric eigenmode benchmarking.
