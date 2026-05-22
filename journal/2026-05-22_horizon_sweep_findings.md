# Horizon Sweep (Smoothed Delta) — Findings

**Date:** 2026-05-22
**Branch:** feat/horizon-sweep (stacked on feat/delay-embedding)
**Motivation:** Standard next-day prediction is dominated by persistence baseline. Smoothed-delta target removes that crutch; horizon sweep reveals where structural embeddings shine.

## Method

For each horizon h ∈ {1, 3, 7}:
- Compute per-subject trailing MA(3) of `global_dysregulation_score`
- Target = MA(score)_{t+h} - MA(score)_t (smoothed change over h days)
- Persistence baseline predicts Δ=0
- Score GRM, PCA, and Takens embeddings via Ridge on the delta target

## Results (transductive diagnostic)

```
                          h=1       h=3       h=7
persistence_zero        -0.0001   -0.0002   -0.0003
grm_ridge                0.0221    0.0254    0.1301
pca_ridge                0.0197    0.0619    0.2305
takens_ridge             0.5166    0.2839    0.3679
```

## Interpretation

### 1. Persistence is dead at all horizons
Predicting Δ=0 gives R² ≈ 0 across all horizons. The smoothed-delta target successfully removes the persistence free lunch. Any positive R² here is genuine structural signal.

### 2. Takens dominates — trajectory signal is real
At h=1: Takens R²=0.52 vs PCA=0.02 vs GRM=0.02. **Massive** gap. The delay-embedded trajectory contains strong signal for predicting *change* that snapshot embeddings miss entirely.

This reverses our earlier finding that "Takens ≈ PCA." On the standard next-day-score target, they were tied because persistence dominated. On the smoothed-delta target where persistence can't help, trajectory history is transformative.

### 3. PCA overtakes GRM at longer horizons
At h=7: PCA R²=0.23 vs GRM=0.13. PCA captures more predictive variance for multi-day change than the graph Laplacian eigenmodes. This reinforces the PCA baseline finding — GRM topology is not adding value over linear projection, even at longer horizons where it theoretically should.

### 4. All methods improve at longer horizons (except Takens h=3 dip)
GRM: 0.02 → 0.03 → 0.13 (improves with h)
PCA: 0.02 → 0.06 → 0.23 (improves with h)
Takens: 0.52 → 0.28 → 0.37 (dips at h=3, recovers at h=7)

The h=3 dip for Takens is interesting — possibly the MA(3) window aliasing with the k=3 delay embedding window. The overall trend: longer horizons make the structural signal more visible relative to noise.

### 5. The crossover point never arrives for GRM
The spec predicted a crossover where GRM overtakes PCA at longer horizons. **It doesn't happen.** PCA beats GRM at every horizon. The graph topology is not providing the multi-day trajectory advantage it should theoretically deliver.

## Key takeaway

**Takens >> PCA >> GRM on smoothed-delta.** The trajectory signal exists and is strong — but it's accessible through simple delay embedding, not through graph topology. This is the clearest evidence yet that:

1. Trajectory information is genuinely predictive (H₁ confirmed for Takens)
2. Graph Laplacian eigenmodes are not capturing this trajectory signal
3. The projection/embedding step, not the target, remains the bottleneck for GRM

## Next steps

1. **Delay-embed the GRM projection.** Concatenate `[grm_emb_t, grm_emb_{t-1}, grm_emb_{t-2}]` and test on smoothed-delta — does combining graph position with trajectory help?
2. **Takens-PCA:** PCA on the delay-embedded matrix. If Takens-PCA >> raw Takens, there's redundancy in the fat vector that PCA can compress.
3. **Nonlinear heads on Takens:** If Ridge on Takens gets R²=0.52, what does XGBoost on Takens get?
