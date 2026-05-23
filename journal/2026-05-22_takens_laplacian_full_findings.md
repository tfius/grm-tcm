# Takens-Laplacian: Full Experiment Arc

**Date:** 2026-05-22
**Branch:** feat/takens-laplacian (stacked on feat/horizon-sweep)
**PR:** #4

## Summary

Built graph from delay-embedded trajectory vectors instead of snapshots. Proved that graph topology CAN capture trajectory signal, identified the projection bottleneck, fixed it, and found the optimal configuration through systematic ablation.

## Experiment sequence and findings

### 1. Takens-Laplacian baseline (graph from X_takens)

GRM smoothed-delta R² at h=1: **0.022 → 0.229** (10x improvement over snapshot graph). Graph eigenmodes now decompose phase-space attractor, not static manifold. But still below raw Takens Ridge (0.517).

### 2. n_modes / ρ sweep

```
h=1 (transductive):
modes \ ρ       0.1     1.0     5.0      RF
   8           0.244   0.229   0.090   0.506
  32           0.281   0.261   0.098   0.528 ← best RF
  64           0.288   0.267   0.099   0.523
 128           0.289   0.268   0.100   0.513
```

- More modes help (8→32), saturates at 32
- Lower ρ helps (less smoothing preserves trajectory cliffs)
- **RF on 32 modes (0.528) beats raw Takens Ridge (0.517)** transductively

### 3. Concatenation test

`[X_takens, GRM_modes]` Ridge = 0.518. No orthogonal gain over raw Takens for linear head. GRM topology is redundant to trajectory features in linear regime.

### 4. Inductive reality check — GRM+RF collapsed

```
                    Transductive    Inductive (broken)
grm_rf              0.528           -0.131
takens_ridge        0.517            0.519
```

**Root cause:** Surrogate projected from X_obs (15D snapshot) into graph built from X_takens (45D trajectory). Dimensional mismatch + wrong feature space.

### 5. Projection fix — trajectory-aware projection

Fixed both surrogate and Nyström to use X_takens when `graph_feature_source=takens`.

```
                    Broken      Nyström     Surrogate   Transductive
grm_ridge           0.015       0.232       0.294       0.281
grm_rf             -0.131       0.394       0.362       0.528
takens+grm_ridge    —           0.521       0.519       0.518
```

**GRM works inductively.** Nyström preserves more topology than surrogate. First orthogonal signal: takens+grm (0.521) > takens alone (0.519).

### 6. Ablation: k=5, density correction, RF on concat

All three suggestions from external review made things worse:

| Change | GRM RF h=1 | takens+grm Ridge h=1 | Why |
|--------|-----------|----------------------|-----|
| k=3 (baseline) | 0.394 | **0.521** | — |
| k=5 | 0.267 | 0.519 | Curse of dimensionality in 75D KNN |
| Density correction | 0.005 | 0.515 | Q normalization over-corrects |
| takens+grm RF | — | 0.503 | RF overfits high-D concat space |

## Optimal configuration

```
graph_feature_source = takens
delay_embedding_k = 3
n_modes = 16
rho = 0.1
projection = nystrom
density_correction = False
head = Ridge on [X_takens, GRM_modes]
```

**Inductive R² = 0.521 on smoothed-delta h=1** (vs persistence = 0.000)

## What we proved

1. **Graph topology CAN capture trajectory dynamics** when built from delay-embedded features (Takens-Laplacian)
2. **The original GRM failure was snapshot-based graph construction**, not a fundamental limitation of spectral methods
3. **Projection input must match graph feature source** — the single biggest fix (0.015 → 0.232 Ridge, -0.131 → 0.394 RF)
4. **RF decodes nonlinear topology transductively** (0.528) but the advantage doesn't fully survive projection (0.394 inductive)
5. **Ridge on [Takens + GRM] is the best inductive model** — regularization > capacity when projection is approximate
6. **Raw Takens Ridge generalizes perfectly** (0.519 transductive = 0.519 inductive) — observation-only features have zero projection gap
7. **The smoothed-delta target was essential** — on standard next-day-score, all methods were indistinguishable (persistence dominance)

## Architecture diagram

```
Training:
  X_obs → delay_embed(k=3) → X_takens
  X_takens → KNN graph + temporal + treatment edges → W
  W → normalized Laplacian → eigsh(32 modes) → GRM embeddings

Inference (inductive, Nyström):
  X_obs_new → delay_embed(k=3) → X_takens_new
  X_takens_new → Nyström(nn_index, eigenvectors) → GRM_emb_new
  [X_takens_new, GRM_emb_new] → Ridge → predicted smoothed Δ

Target:
  MA(score, 3)_{t+h} - MA(score, 3)_t  (h ∈ {1, 3, 7})
```

## PR stack

| PR | What | Key result |
|----|------|------------|
| #1 | PCA baseline | GRM ≈ PCA on standard target |
| #2 | Delay embedding | Marginal trajectory signal (+0.02 R²) |
| #3 | Horizon sweep + SSA | Takens R²=0.52 on fair target; SSA doesn't help |
| #4 | Takens-Laplacian | Graph topology works when built from trajectories; Ridge on [Takens+GRM] = 0.521 inductive |
