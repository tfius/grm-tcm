# Takens-Laplacian — Findings

**Date:** 2026-05-22
**Branch:** feat/takens-laplacian (stacked on feat/horizon-sweep)
**Motivation:** Build the graph from delay-embedded trajectory vectors instead of snapshots, so Laplacian eigenmodes decompose the phase-space attractor rather than the static symptom manifold.

## Method

New config `graph_feature_source = "takens"`:
- `_build_visit_graph` receives X_takens (delay-embedded, dim=p*k) instead of X_obs (dim=p)
- KNN edges now encode "similar symptoms AND similar recent history"
- Temporal and treatment edges unchanged
- σ (RBF bandwidth) auto-scales via median heuristic — no manual adjustment needed

## Results (transductive diagnostic)

### Horizon sweep — GRM with Takens graph vs obs graph

```
                          h=1       h=3       h=7
GRM (obs graph)          0.022     0.025     0.130
GRM (takens graph)       0.229     0.150     0.199
raw Takens Ridge         0.517     0.284     0.368
```

**GRM improves 10x at h=1** when the graph is built from trajectories (0.022 → 0.229). Still below raw Takens Ridge (0.517), but the gap closed from 24x to 2.3x.

### Standard targets (next_day_score)

| Model | obs graph | takens graph |
|-------|-----------|--------------|
| grm_ridge R² | 0.238 | 0.229 |
| grm+lag R² | 0.564 | 0.561 |
| grm AUC | 0.621 | **0.633** |
| grm+lag AUC | 0.654 | **0.656** |
| Aliased AUC | 0.664 | **0.664** |

Standard regression slightly worse (noise from wider feature space in KNN). Classification slightly better — trajectory-aware graph helps distinguish flare risk.

### Aliased subset

| Model | obs graph | takens graph |
|-------|-----------|--------------|
| grm+lag R² | 0.445 | 0.442 |
| grm+lag AUC | 0.664 | **0.664** |

No change on aliased subset — the trajectory information doesn't help disambiguate the aliased observation pairs on standard targets.

## Interpretation

1. **Takens-Laplacian dramatically improves smoothed-delta prediction.** The graph eigenmodes now capture trajectory dynamics that were invisible to the snapshot graph. This validates the theoretical argument: KNN on trajectories → phase-space attractor decomposition.

2. **Still below raw Takens Ridge.** R²=0.23 vs 0.52. The spectral filter `g(λ) = 1/(1+ρ²λ)` is compressing the trajectory-enriched graph into 8 modes, losing signal. Raw Takens with 45 dimensions and Ridge has more capacity.

3. **Standard targets don't show the improvement** because persistence still dominates. The horizon sweep / smoothed-delta target is essential for revealing trajectory value.

4. **Classification benefits modestly** — AUC improves from 0.621 to 0.633. The trajectory graph helps with flare prediction even on the standard target.

## What this means

The Takens-Laplacian proves that **graph topology CAN capture trajectory signal when the graph is built from the right features.** The original GRM failure wasn't a fundamental limitation of spectral methods — it was the snapshot-based graph construction.

Remaining gap (0.23 vs 0.52) is likely due to:
- Too few modes (n_modes=8 compresses 45D trajectory space aggressively)
- ρ=1.0 may over-smooth trajectory-enriched eigenmodes
- Linear Ridge on 8 GRM modes vs 45 Takens features — capacity gap

## Next steps

1. **n_modes sweep:** Try n_modes ∈ {8, 16, 24, 32} with Takens-Laplacian
2. **ρ sweep:** Try ρ ∈ {0.1, 0.5, 1.0, 2.0} with Takens graph
3. **Takens-Laplacian + raw Takens concatenation:** `[grm_takens_emb, x_takens]` — graph position plus raw trajectory
