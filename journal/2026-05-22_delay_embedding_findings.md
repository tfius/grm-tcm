# Delay-Embedded Surrogate (Takens) Baseline — Findings

**Date:** 2026-05-22
**Branch:** feat/delay-embedding (stacked on feat/pca-baseline)
**Motivation:** Before building a complex graph OOSE (Out-Of-Sample Extension) for trajectory projection, test whether historical trajectory contains *any* predictive signal via cheap delay embedding (Takens' Theorem).

## Hypotheses

- **H₀:** Delay-embedded trajectory `[x_t, x_{t-1}, x_{t-2}]` yields no improvement over snapshot `x_t`. Observation noise destroys trajectory signal.
- **H₁:** Delay embedding significantly outperforms static baselines, especially on aliased-pair subset. Historical velocity resolves identical-looking surface symptoms.

## Method

- `_build_delay_embedding(X, visits, k=3)`: concatenates last k visits per subject into a fat vector (shape N x p*k). Respects subject boundaries; early visits padded with NaN → median imputed.
- Same Ridge/Logistic heads and lag augmentation as GRM and PCA.
- Evaluated on: regression, classification, flare onset (full + hard), aliased-pair subset.

## Results (transductive diagnostic)

### Full dataset comparison

| Task | GRM+lag | PCA+lag | Takens+lag | Winner |
|------|---------|---------|------------|--------|
| Regression R² | 0.5643 | 0.5789 | **0.5828** | Takens |
| Classification AUC | 0.6543 | 0.6536 | 0.6534 | ~tie |
| Onset AUC (hard) | **0.6211** | 0.6195 | 0.6130 | GRM |
| Aliased R² | 0.4446 | 0.4510 | **0.4562** | Takens |
| Aliased AUC | **0.6636** | 0.6621 | 0.6631 | GRM |

Without lag features:

| Task | GRM | PCA | Takens | Winner |
|------|-----|-----|--------|--------|
| Regression R² | 0.2379 | 0.3635 | **0.3860** | Takens |
| Classification AUC | 0.6210 | 0.6239 | **0.6255** | Takens |

## Interpretation

**Mixed signal — H₀ mostly holds with a faint trajectory advantage.**

1. **Takens shows small but consistent regression lift.** R²=0.3860 vs PCA=0.3635 vs GRM=0.2379 (without lag). The trajectory window adds ~+0.02 R² over PCA snapshot. Real but small.

2. **Classification is a wash.** AUC differences are within noise margins (0.6255 vs 0.6239 vs 0.6210). Trajectory doesn't help predict binary flare status.

3. **Aliased subset: Takens wins regression, GRM wins classification.** Takens R²=0.4562 vs PCA=0.4510 vs GRM=0.4446. But GRM AUC=0.6636 beats Takens AUC=0.6631. No method dominates.

4. **Hard onset: Takens actually worse.** AUC=0.6130 vs GRM=0.6211 vs PCA=0.6195. The trajectory window is *hurting* onset prediction — possibly because the wider feature vector overfits or the median-imputed early visits add noise.

5. **Lag features still dominate everything.** All methods jump from ~0.35-0.39 to ~0.57-0.58 when yesterday's score is appended. The +lag signal dwarfs the takens-vs-snapshot difference.

## Go/No-Go verdict

**Marginal go — trajectory signal exists but is weak.** The k=3 delay embedding adds ~0.02 R² over static PCA on regression. Not enough to justify a multi-week graph OOSE refactor.

Recommended: try k ∈ {2, 5, 7} to find saturation point before deciding. If lift plateaus at k=3, the trajectory signal ceiling is low and the OOSE investment is not justified.

## What this means combined with PCA findings

Two baselines now established:
1. **PCA ≈ GRM** → graph topology not adding value over linear projection
2. **Takens ≈ PCA** → trajectory history barely improves over snapshot

The remaining levers:
- **Smoothed-delta target** — change what we predict (remove persistence dominance)
- **Horizon sweep** — change prediction horizon (find where trajectory matters)
- **Nonlinear heads** — change prediction model (test if structure exists but Ridge can't see it)
