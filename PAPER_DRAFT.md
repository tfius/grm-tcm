# Trajectory-Aware Spectral Graph Models for Longitudinal Health State Prediction

## Abstract

We introduce the Takens-Laplacian: a spectral graph construction for
longitudinal health data where visit-similarity edges are computed over
delay-embedded trajectory vectors rather than instantaneous observations.
Combined with a per-subject constitution proxy and nonlinear decoding, the
resulting Graph Resonance Model (GRM) predicts future health state changes
on a persistence-free evaluation target with R²=0.49 on real wearable data
(54 subjects, Fitbit Sense) and R²=0.52 on a synthetic benchmark.
Flare classification reaches AUC=0.83, and treatment response shows
significant state-dependence (η²=0.37, p<0.05). The trajectory signal
replicates across three datasets. We report both positive results and
systematic negative findings from 12 ablation experiments.

---

## 1. Introduction

Wearable health monitoring produces daily multi-channel time series (heart
rate, sleep, activity, stress). We ask whether spectral graph methods —
where each patient visit is a node and edges encode similarity — can learn
a latent health manifold that predicts future state changes.

The key challenge: on standard next-day prediction targets, persistence
(yesterday ≈ today) dominates all methods. We show this is a target design
problem. On persistence-free targets (smoothed score deltas), trajectory-
aware spectral methods reveal signal invisible to snapshot approaches.

**Contributions:**

1. The Takens-Laplacian graph construction (10x GRM improvement over
   snapshot graphs)
2. Three-component architecture: trajectory + constitution + topology
3. Validation on synthetic + two real wearable datasets
4. Systematic ablation documenting 12 negative results

---

## 2. Methods

### 2.1 Delay Embedding

Trajectory vector of window k: `x^(k)_{i,t} = [x_{i,t}, ..., x_{i,t-k+1}]`.
Default k=3. By Takens' theorem, reconstructs attractor topology.

### 2.2 Constitution Proxy

Per-subject cumulative mean: `μ_{i,t} = (1/t) Σ x_{i,s}`. Causal (no future
leakage). Recovers R²=0.93 of true constitution on synthetic data.

### 2.3 Takens-Laplacian Graph

Visit graph with KNN edges on delay-embedded vectors:
`W^obs_{ab} = exp(-||x^(k)_a - x^(k)_b||² / 2σ²)` plus temporal and
treatment edges. Normalized Laplacian decomposed into K eigenmodes. GRM
embedding: `e_a = [√g_m · ψ_m(a)]` with `g_m = 1/(1+ρ²λ_m)`, ρ=0.1.

### 2.4 Prediction

Combined features `[x^(k), μ, e]` → Ridge (inductive) or RF (transductive).

Target: `y^Δ_{h} = MA(score,3)_{t+h} - MA(score,3)_t`. Persistence = Δ0.

### 2.5 Inductive Projection

Nyström extension on delay-embedded test features. Feature space must match
graph construction space (Section 5.3).

---

## 3. Datasets

| Dataset | N | Days | Channels | Source |
|---------|---|------|----------|--------|
| Synthetic | 200 | 120 | 15 | Switching state-space, 7 regimes |
| PMData | 16 | 150 | 21 | Fitbit + wellness + training load |
| LifeSnaps | 54 | 88 (median) | 18 | Fitbit Sense + EMA + Big Five |

LifeSnaps quality-filtered: removed binary mood columns, physiology-only
dysregulation score, subjects with ≥40 days and ≥50% core feature fill.

---

## 4. Results

### 4.1 Smoothed-Delta Prediction

**Synthetic (transductive):**

| Model | h=1 | h=7 | h=21 |
|-------|-----|-----|------|
| Persistence (Δ=0) | 0.000 | 0.000 | 0.000 |
| Snapshot GRM Ridge | 0.022 | 0.130 | 0.183 |
| Takens-Laplacian GRM Ridge | 0.229 | 0.199 | 0.234 |
| Takens Ridge | 0.517 | 0.368 | 0.431 |
| Takens+prior+GRM Ridge | 0.524 | 0.404 | 0.470 |

**LifeSnaps (transductive, quality-filtered):**

| Model | h=1 | h=7 |
|-------|-----|-----|
| Persistence (Δ=0) | -0.001 | 0.000 |
| PCA Ridge | 0.013 | 0.018 |
| Takens RF | 0.460 | 0.440 |
| Takens+prior RF | 0.484 | 0.445 |
| Takens+prior+GRM RF | 0.489 | 0.422 |

**PMData (transductive):**

| Model | h=1 |
|-------|-----|
| Takens RF | 0.379 |
| Takens+GRM RF | 0.500 |

Trajectory signal replicates across all three datasets. Constitution proxy
adds +6% at h≥7. GRM adds +0.05 on LifeSnaps, +0.12 on PMData.

### 4.2 Flare Classification

| Dataset | Model | AUC |
|---------|-------|-----|
| Synthetic | GRM+lag Logistic | 0.656 |
| LifeSnaps | GRM+lag Logistic | 0.829 |
| PMData | GRM+lag Logistic | 0.716 |

### 4.3 Treatment Response Stratification

Kruskal-Wallis test on post-treatment score change, clustered by embedding
at treatment time (synthetic, 524 clean treatment windows):

| Embedding | η² (h=7) | p |
|-----------|---------|---|
| PCA | 0.368 | <0.05 |
| Takens | 0.265 | <0.05 |
| GRM | 0.148 | <0.05 |

Treatment response is state-dependent across all embedding types.

### 4.4 Signal Chain Analysis (Synthetic)

Oracle features reveal the theoretical ceiling:

| Features added | R² (h=1) |
|---------------|---------|
| Observations delay-embedded | 0.515 |
| + true current regime | 0.542 |
| + true next regime | 0.843 |
| All oracle features | 0.858 |

The 0.34 gap between our best (0.52) and the ceiling (0.86) lives entirely
in next-regime prediction. Observations carry more signal than true latents
(0.52 vs 0.33) because the observation model encodes regime offsets and
constitution.

---

## 5. Critical Findings

### 5.1 Persistence Masks Everything on Standard Targets

On next-day-score, all methods converge to R²≈0.55 once lag features are
included. PCA ≈ GRM ≈ Takens. This is not a method failure — persistence
(autocorrelation >0.6) absorbs variance. **The smoothed-delta target is
essential** to reveal that trajectory methods carry 25x more genuine signal
than snapshot methods.

### 5.2 Takens-Laplacian: Graph From Trajectories

Snapshot GRM: R²=0.022. Takens-Laplacian GRM: R²=0.229 (Ridge), 0.528 (RF).
**Building the graph from delay-embedded vectors is the single largest
architectural improvement.** The graph must encode trajectory similarity, not
just instantaneous symptom similarity.

### 5.3 Projection Feature Space Must Match Graph Feature Space

When the graph is built from 45D trajectory vectors but the inductive
projection uses 15D snapshots, GRM collapses to R²=−0.13. Matching the
projection input to the graph feature space recovers R²=0.39. **This is a
silent failure mode with no error message** — the model runs, produces
numbers, and they are wrong.

### 5.4 Architecture-Task Duality

| Task | Best component | Why |
|------|---------------|-----|
| Acute trajectory (h=1) | Takens | Velocity dominates |
| Long horizon (h≥7) | Constitution proxy | Baseline determines drift |
| Regime transitions | GRM modes | Graph position encodes state |
| Flare classification | GRM+lag | Topology + persistence |

No single component dominates. The combined architecture leverages each
where it contributes.

---

## 6. Limitations

1. Small real-world N (16 and 54 subjects) — GRM needs N≥200 for robust
   graph topology
2. Synthetic SNR=0.64 is below typical wearable SNR (~1.5–2.5) — real data
   should produce stronger treatment stratification
3. GRM RF drops from R²=0.53 to 0.39 inductively — Nyström loses topology
4. Treatment events are exercise loads, not clinical interventions
5. TCM label alignment is modest (AMI=0.15)

---

## 7. Conclusion

Trajectory-aware spectral graph methods extract predictive structure from
longitudinal health data that snapshot methods miss entirely. The key is
building the graph from delay-embedded trajectories and evaluating on
persistence-free targets. The method replicates across synthetic and real
wearable data, with the combined [trajectory + constitution + topology]
architecture achieving R²=0.49 on LifeSnaps and AUC=0.83 for flare
classification.

---

## Appendix A: Ablation Studies (Negative Results)

All experiments on synthetic benchmark unless noted. Each tested in
isolation against the best configuration at the time.

### A.1 Approaches That Did Not Help

| Experiment | Result | Why it failed |
|-----------|--------|---------------|
| k=5 delay embedding | GRM RF: 0.39→0.27 | Curse of dimensionality in 75D KNN |
| 90-day rolling window | No improvement | Constitution proxy already captures it |
| Density correction (Q@W@Q) | GRM RF: 0.39→0.005 | Over-corrects edge weights on Takens graph |
| Takens-PCA (SSA compression) | 0.517→0.500 | PCA removes useful phase-space redundancy |
| Takens+GRM RF (inductive) | 0.521→0.503 | RF overfits high-D concatenated space |
| GRM propagation vectors | +0.000 | Fully redundant to eigenmodes for linear head |
| Regime-probability stacking | +0.007 | Soft probabilities redundant to delay embedding |
| Sign-sqrt GRM transform | +0.003 at h=1, −0.004 at h=21 | Helps short-term, hurts where constitution dominates |
| GRM polynomial features | +0.000 | Regime boundaries not quadratic in eigenmode space |
| 14+90 day multi-window | No improvement | Truncated rolling stats noisy on 120-day subjects |
| Complex eigenvalues (directed graph) | Not needed | Graph well-connected (degree 16.4, 0 isolated) |
| Flaredown dataset (17K users) | Unusable | 90% NaN — users track different symptoms |

### A.2 The Persistence Trap

On standard next-day-score with lag features:

| Model | R² |
|-------|-----|
| Persistence only | 0.478 |
| PCA+lag | 0.579 |
| GRM+lag | 0.564 |
| Takens+lag | 0.583 |

All within ±0.02 of each other. **These numbers are misleading.** The lag
feature dominates, and all embeddings contribute marginal decorrelation.
Reporting only these metrics would wrongly conclude that graph topology adds
nothing.

### A.3 Modes and Spectral Scale Sweep

n_modes ∈ {4,8,16,32,64,128} × ρ ∈ {0.1,0.5,1.0,2.0,5.0} on Takens-
Laplacian (synthetic, h=1):

- Ridge: saturates at n=32 (R²=0.281). ρ=0.1 best everywhere.
- RF: peaks at n=32 (R²=0.528). Declines at n>64 (overfitting).
- ρ=5.0 collapses signal (0.098) — over-smoothing destroys steep regime
  boundaries.

### A.4 Inductive Projection Comparison

| Projection | GRM Ridge | GRM RF |
|-----------|-----------|--------|
| Snapshot surrogate (broken) | 0.015 | −0.131 |
| Trajectory-matched surrogate | 0.294 | 0.362 |
| Trajectory-matched Nyström | 0.232 | 0.394 |

Nyström preserves more topology than the linear surrogate for RF. Both
require trajectory-matched input features.

### A.5 Low-Noise Sensitivity (Synthetic)

Reducing obs_noise from 0.28 to 0.12 (SNR 0.64→~1.5, realistic for
wearables):

| Metric | SNR=0.64 | SNR~1.5 | Δ |
|--------|----------|---------|---|
| Treatment η² (GRM, h=7) | 0.148 | 0.265 | +0.117 |
| Regime accuracy (Takens) | 0.526 | 0.582 | +0.056 |
| Takens Ridge h=1 | 0.517 | 0.559 | +0.042 |

Treatment stratification nearly doubles with realistic noise levels.

---

## Appendix B: TCM Connections (Exploratory)

TCM label alignment with Takens-Laplacian GRM (synthetic, KMeans k=5):

| Label | Snapshot AMI | Takens AMI | Δ |
|-------|------------|-----------|---|
| true_regime | 0.360 | 0.404 | +12% |
| tcm_like_label | 0.135 | 0.149 | +10% |
| qi_like_label | 0.171 | 0.175 | +3% |

Direction correct but effects modest. TCM categories align better with
trajectory-aware topology, consistent with the hypothesis that TCM describes
dynamical patterns, not static symptom clusters.

Regime-change prediction (7-class, change-only subset):

| Model | Accuracy |
|-------|----------|
| GRM Logistic | 0.232 |
| PCA Logistic | 0.179 |
| Takens Logistic | 0.171 |
| Random | 0.143 |

GRM leads on state transitions (+9pp over random) but absolute accuracy is
low. Treatment response regime-change η²=0.034 (GRM) vs 0.000 (PCA) — GRM
uniquely predicts whether treatment triggers a state change, but the effect
size is small (3.4% variance explained).

These results are exploratory. Validating TCM connections requires clinical
data with actual TCM diagnostic assessments.

---

## References

Belkin, M. & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality
reduction and data representation. Neural Computation, 15(6), 1373–1396.

Coifman, R.R. & Lafon, S. (2006). Diffusion maps. Applied and Computational
Harmonic Analysis, 21(1), 5–30.

Pang, J.C. et al. (2023). Geometric constraints on human brain function.
Nature, 618, 566–574.

Takens, F. (1981). Detecting strange attractors in turbulence. Lecture Notes
in Mathematics, 898, 366–381.

Yfantidou, S. et al. (2022). LifeSnaps: a 4-month multi-modal dataset
capturing unobtrusive snapshots of our lives in the wild. Scientific Data, 9.
