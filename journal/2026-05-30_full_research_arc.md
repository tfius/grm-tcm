# GRM-TCM Research Arc — Complete Summary

**Date:** 2026-05-22 through 2026-05-30
**PRs:** #1–#13

## The Question

Can a Graph Resonance Model (GRM) learn predictive latent structure from
longitudinal health observations, and does it add value beyond simpler
baselines?

## What We Built

A spectral graph pipeline that takes daily health observations, constructs
a visit-similarity graph, decomposes it into eigenmodes, and uses those
modes alongside trajectory features to predict future health state changes.

Tested on one synthetic benchmark and two real-world wearable datasets.

---

## Phase 1: Establishing Baselines (PRs #1–#2)

### PCA Baseline (#1)
- PCA(n_modes) on raw observations → Ridge/Logistic
- **Result: GRM ≈ PCA on standard next-day-score target**
- Lag features (yesterday's value) dominate both embeddings (+0.20 R²)
- Graph topology adds nothing over linear variance extraction

### Delay Embedding (#2)
- Takens delay embedding (k=3): concatenate [x_t, x_{t-1}, x_{t-2}]
- **Result: marginal gain (+0.02 R²) on standard target**
- Trajectory signal exists but persistence masks it

**Lesson: the standard next-day target is persistence-dominated. Need a
fairer evaluation target to see real differences.**

---

## Phase 2: The Breakthrough (PRs #3–#4)

### Smoothed-Delta Target + Horizon Sweep (#3)
- Target = MA(score,3)_{t+h} - MA(score,3)_t at h=1,3,7
- Persistence predicts Δ=0 → any positive R² is genuine signal
- **Result: Takens R²=0.52 vs GRM=0.02 vs PCA=0.02**
- Trajectory signal is 25x more predictive than either GRM or PCA
- The "PCA ≈ GRM ≈ Takens" finding was an artifact of persistence

### Takens-Laplacian (#4)
- Build graph from delay-embedded trajectory vectors, not snapshots
- **Result: GRM jumps from R²=0.02 to 0.23 (Ridge), 0.53 (RF)**
- Graph topology CAN capture trajectory signal when built from the
  right features
- RF on 32 modes beats raw Takens Ridge (0.528 vs 0.517)

### Inductive Projection Fix (#4)
- Discovered: surrogate projected from X_obs (15D) into graph built
  from X_takens (45D) → complete collapse (R²=-0.13)
- **Fix: match projection feature space to graph feature space**
- Nyström on X_takens: GRM RF recovers to R²=0.39 inductively
- This was the single biggest fix in the entire project

**Lesson: the graph feature space and the projection feature space
MUST match. Mismatch is a silent failure mode.**

---

## Phase 3: Architecture Refinement (PRs #5–#8)

### TCM Alignment (#5)
- KMeans on GRM modes → AMI/ARI/NMI against TCM labels
- Trajectory-aware graph: true_regime AMI 0.360→0.404 (+12%)
- TCM labels align better with trajectory topology than snapshot
- Effect is real but modest (AMI ~0.15)

### Multi-Scale Features (#6)
- 14-day rolling mean/std/slope + k=3 trajectory
- Helps at h=7 (+4% R²), not at h=1
- Chronic baseline patterns matter for multi-day prediction

### Regime Transitions (#7)
- 7-class next-regime prediction
- GRM logistic leads on regime-CHANGE subset (0.232 vs 0.171)
- Graph position predicts which-state-next better than velocity
- But 23% accuracy on 7 classes is modest (random=14%)

### Treatment Response (#8)
- Kruskal-Wallis test: does embedding cluster predict treatment outcome?
- **η²=0.37 (PCA, h=7) — large effect, genuinely significant**
- Treatment response IS state-dependent
- GRM uniquely captures regime-change response (η²=0.034 — small)

**Lesson: different tasks favor different architectures. Trajectory for
score prediction, graph position for regime transitions, PCA for
treatment stratification.**

---

## Phase 4: Signal Chain Analysis (PR #10+)

### Theoretical Ceiling Discovery
- Oracle features analysis revealed: knowing next-regime = +0.30 R²
- Our best (takens+prior): R²=0.52. Ceiling: R²=0.86.
- The 0.34 gap is entirely in regime transition prediction
- Observations contain MORE signal than true latents (R²=0.52 vs 0.33)
  because the observation model encodes regime offsets + constitution

### Constitution Proxy (PR #11)
- Per-subject cumulative mean of all prior observations
- **h=7: +6% R² (0.381→0.404). h=21: peaks at R²=0.470**
- Stable patient baseline is the biggest lever at long horizons
- R² increases with horizon up to h=21: "constitutional medicine" signature

### Generator Noise Analysis (PR #12)
- Current synthetic SNR=0.64 — worse than real wearable data (~1.5-2.5)
- Low noise (obs=0.12): treatment η² doubles (0.15→0.27)
- Real data should produce stronger treatment stratification
- Sign-sqrt GRM transform: +0.003 R² at h=1 (marginal)

### Ablations (Negative Results)
- k=5 delay embedding: hurts (curse of dimensionality in KNN)
- 90-day rolling window: doesn't help (constitution proxy already captures it)
- Density correction (Q@W@Q): destroys signal (over-corrects)
- Takens+GRM RF concat: overfits (Ridge beats RF on high-D concat)
- Takens-PCA (SSA): loses signal (redundancy is useful, not noise)
- GRM propagation vectors (G_to_anchors): redundant to eigenmodes
- Regime-probability stacking: redundant to delay embedding
- Complex eigenvalues: not needed (graph is well-connected, not shattered)

---

## Phase 5: Real-World Validation (PR #13)

### PMData (16 athletes × 150 days, Fitbit)
- **Takens Ridge R²=0.36 on smoothed-delta h=1**
- GRM+lag AUC=0.72 for flare classification
- Trajectory signal replicates on real human wearable data
- GRM modes weak (R²=0.006) — 16 subjects too few for graph

### LifeSnaps (54 subjects × 88 days median, Fitbit Sense)
- Quality-filtered: removed binary mood columns, physiology-only dysregulation
- Hourly data → HR variability, circadian amplitude features
- **Best: takens+prior+grm RF R²=0.489 at h=1**
  - Constitution proxy: +0.06 (genuine individual baseline effect)
  - GRM topology: +0.05 (first clean evidence on real data)
  - Hourly features: +0.06 (HR variability is real physiological signal)
- GRM logistic AUC=0.83 for flare classification

### Datasets That Didn't Work
- **Flaredown (17K autoimmune users):** 90% NaN — each user tracks
  different symptoms, no common feature space
- **GlucoBench:** 10 users × 9 days, probably synthetic
- **GLOBEM:** needs PhysioNet credentials (CITI training)

---

## Final Architecture

```
Input:    [x_t, x_{t-1}, x_{t-2}]              Takens trajectory (k=3)
        + per_subject_cumulative_mean(x)        Constitution proxy
        + GRM_modes(Takens-Laplacian graph)     Spectral topology

Graph:    KNN on delay-embedded vectors + temporal + treatment edges
          σ auto-scales via median heuristic
          n_modes=8-16, ρ=0.1

Head:     RandomForest (real data) or Ridge (synthetic/inductive)

Target:   MA(score,3)_{t+h} - MA(score,3)_t     Smoothed delta

Projection: Nyström extension matching graph feature space
```

## Results Across All Datasets

| Dataset | N | Days | Best R² h=1 | Model |
|---------|---|------|-------------|-------|
| Synthetic | 200 | 120 | 0.524 | takens+prior+grm Ridge |
| PMData | 16 | 150 | 0.379 | takens RF |
| LifeSnaps | 54 | 88 | 0.489 | takens+prior+grm RF |

## What We Proved

1. **Trajectory signal is real and strong** across synthetic and real data
2. **Graph topology adds genuine value** when built from trajectories
   (Takens-Laplacian) and decoded nonlinearly (RF)
3. **Constitution proxy improves long-horizon prediction** — stable patient
   baseline determines multi-week trajectory
4. **Treatment response is state-dependent** (η²=0.37, p<0.05)
5. **Projection must match graph feature space** — biggest silent failure mode
6. **Standard next-day targets hide trajectory signal** — need persistence-free
   evaluation (smoothed delta)
7. **TCM labels align better with trajectory-aware topology** than snapshots

## What We Didn't Prove

1. GRM topology didn't survive inductive projection well (0.53→0.39)
2. Regime transitions remain hard to predict (23% on 7 classes)
3. TCM alignment is modest (AMI ~0.15)
4. Treatment response stratification — PCA beats GRM for score-delta
5. On real data, GRM adds +0.05 R² — helpful but not transformative

## What Would Move the Needle Next

1. **Larger real-world dataset (N≥200)** with daily wearable physiology —
   GRM graph construction needs mass to build meaningful topology
2. **Real treatment data** — clinical interventions, not just exercise load
3. **Longer follow-up (6+ months)** — constitution effects need time
4. **Nonlinear inductive projection** — small MLP surrogate from X_takens
   to GRM embeddings, properly regularized
5. **Domain-specific features** — circadian rhythm metrics, HRV time-domain
   and frequency-domain features from raw sensor data
