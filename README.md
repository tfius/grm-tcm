# GRM-TCM Synthetic Benchmark

Graph Resonance Model experiments for longitudinal synthetic whole-body state
data, TCM-like semantic labels, and explicit manifold-geometry controls.

This is a research benchmark. It does not validate TCM, Qi, or a biological
mechanism. The project asks narrower questions:

- Can graph spectral/resonance features improve future-state prediction?
- When do GRM modes recover known synthetic latent structure?
- Do TCM-like labels align with learned states, or do they merge/split true regimes?
- Can a visit graph recover a known Laplace-Beltrami basis on a synthetic body-state manifold?

This project was explored at [muShanghai](https://mushanghai.xyz/), a Shanghai
builder event/program focused on AI, biotech/longevity, robotics/hardware, and
culture.

## Setup

```bash
uv sync
```

Run tests:

```bash
uv sync --group dev
uv run pytest
```

## Quick Start: Regime Benchmark

The default synthetic generator is the clinical/regime benchmark. It creates
200 subjects x 120 days by default.

```bash
uv run python grm_tcm_synthetic_generator.py
uv run python grm_tcm_train.py
uv run python grm_tcm_diagnostics.py
uv run python grm_tcm_dynamic_grm.py
```

Useful outputs:

```text
synthetic_grm_tcm/
  subjects.csv
  visits.csv
  latent_states.csv
  events.csv
  true_regimes.csv
  true_transition_matrices.csv
  true_attractor_states.csv
  metadata.json

grm_tcm_results/
  grm_visit_embeddings.csv
  grm_feature_modes.csv
  grm_predictions.csv
  grm_metrics.json
  model/

grm_tcm_diagnostics/
  diagnostics_summary.json
  cluster_scores.csv
  contrarian_findings.csv
  ontology_mismatch.csv
  plots/

grm_tcm_dynamic/
  dynamic_grm_metrics.json
  rolling_regime_scores.csv
  spectral_energy.csv
  self_resonance_scores.csv
  subject_resonance_summary.csv
  grm_transition_predictions.csv
  transition_reliability.csv
  plots/
  model/
```

Override scale:

```bash
uv run python grm_tcm_synthetic_generator.py --n-subjects 200 --n-days 120
```

## Static Training

Default static training:

```bash
uv run python grm_tcm_train.py
```

The default graph mode is:

```text
feature_temporal_treatment
```

It combines observation KNN edges, same-subject temporal edges, and treatment
context edges. This is the main prediction graph.

Strict held-out-subject evaluation:

```bash
uv run python grm_tcm_train.py --inductive \
    --projection surrogate \
    --output-dir grm_tcm_results_inductive
```

Important graph modes:

```bash
# Observation KNN only.
uv run python grm_tcm_train.py --graph-mode feature_only

# Geometry-recovery mode: observation KNN with diffusion-map density correction.
uv run python grm_tcm_train.py --graph-mode feature_only_diffusion --diffusion-alpha 1.0

# Prediction default.
uv run python grm_tcm_train.py --graph-mode feature_temporal_treatment

# Experimental constitution/subject-similarity graph.
uv run python grm_tcm_train.py --graph-mode feature_temporal_treatment_subject
```

Static `grm_mode_*` coordinates use the kernel-feature convention:

```text
sqrt(1 / (1 + rho^2 lambda_k)) * psi_k
```

so inner products of saved embeddings reconstruct the truncated GRM kernel.
This convention starts at static model schema `static-v3`; retrain older saved
models before using `predict.py`.

## Static Diagnostics

After static training:

```bash
uv run python grm_tcm_diagnostics.py
```

Useful static plots:

- `grm_tcm_diagnostics/plots/grm_latent_correlation_heatmap.png`
- `grm_tcm_diagnostics/plots/predicted_vs_actual_next_day_score.png`
- `grm_tcm_diagnostics/plots/residual_histogram.png`
- `grm_tcm_diagnostics/plots/grm_modes_scatter_hidden_subtype.png`
- `grm_tcm_diagnostics/plots/grm_modes_scatter_true_regime.png`
- `grm_tcm_diagnostics/plots/grm_modes_scatter_tcm_like_label.png`
- `grm_tcm_diagnostics/plots/true_regime_distribution_by_tcm_label.png`

Read first:

```bash
cat grm_tcm_diagnostics/diagnostics_summary.json
```

## Dynamic GRM

Run dynamic state/resonance diagnostics after static training:

```bash
uv run python grm_tcm_dynamic_grm.py
```

Useful options:

```bash
uv run python grm_tcm_dynamic_grm.py --similarity-mode knn --state-similarity-k 3
uv run python grm_tcm_dynamic_grm.py --similarity-mode threshold --similarity-quantile 0.8
uv run python grm_tcm_dynamic_grm.py --state-fit-end-day 40
uv run python grm_tcm_dynamic_grm.py --state-source kmeans_dynamic
uv run python grm_tcm_dynamic_grm.py --state-source true_regime
uv run python grm_tcm_dynamic_grm.py --compare-state-sources
```

State sources:

- `kmeans_observation` tests visible observation clusters.
- `kmeans_dynamic` adds short trajectory features to reduce aliasing.
- `true_regime` is an oracle ceiling for the synthetic benchmark.

Useful dynamic plots:

- `grm_tcm_dynamic/plots/rolling_regime_change_score.png`
- `grm_tcm_dynamic/plots/self_resonance_vs_dysregulation.png`
- `grm_tcm_dynamic/plots/soft_self_resonance_vs_dysregulation.png`
- `grm_tcm_dynamic/plots/pooled_transition_reliability.png`
- `grm_tcm_dynamic/plots/inferred_state_true_regime_confusion.png`
- `grm_tcm_dynamic/plots/subject_resonance_vs_true_stuck_occupancy.png`
- `grm_tcm_dynamic/plots/cumulative_spectral_energy.png`
- `grm_tcm_dynamic/plots/state_source_metric_comparison.png`

## Falsifiable Dynamic Eval

After static + dynamic models exist:

```bash
uv run python grm_tcm_dynamic_eval.py \
    --static-model-dir grm_tcm_results/model \
    --dynamic-model-dir grm_tcm_dynamic/model
```

This writes:

```text
grm_tcm_dynamic_eval/dynamic_eval_certificates.json
grm_tcm_dynamic_eval/transition_metrics.csv
grm_tcm_dynamic_eval/aliased_state_analysis.csv
grm_tcm_dynamic_eval/ablation_metrics.csv
grm_tcm_dynamic_eval/plots/
```

Run only selected scopes:

```bash
uv run python grm_tcm_dynamic_eval.py --scope transitions,aliased,ablations
uv run python grm_tcm_dynamic_eval.py --scope plots
```

The eval uses bootstrap confidence intervals and explicit controls such as
random embeddings, shuffled time, raw observations, and Markov baselines.

## Manifold Geometry Benchmark

The regime generator tests clinical-state prediction. The manifold generator
tests whether GRM recovers a known geometry.

Generate the manifold data:

```bash
uv run python grm_tcm_manifold_generator.py
```

It writes:

```text
synthetic_grm_tcm_manifold/
  subjects.csv
  visits.csv
  latent_states.csv
  events.csv
  true_lbo_modes.csv
  true_lbo_eigenmodes.npz
  metadata.json
```

The synthetic ground truth is a 2D torus. The true Laplace-Beltrami modes are
analytical Fourier/LBO modes saved as `true_lbo_mode_*` columns and in
`true_lbo_eigenmodes.npz`.

Train static GRM on manifold data:

```bash
uv run python grm_tcm_train.py \
    --input-dir synthetic_grm_tcm_manifold \
    --output-dir grm_tcm_results_manifold
```

Evaluate geometry recovery:

```bash
uv run python grm_tcm_manifold_eval.py \
    --data-dir synthetic_grm_tcm_manifold \
    --results-dir grm_tcm_results_manifold
```

The evaluator compares:

- `oracle_torus_diffusion`: graph built from true torus coordinates.
- `observation_diffusion`: graph built from observed channels only.
- `saved_static_grm`: embeddings from `grm_tcm_train.py`.

It scores eigenspaces, not only individual modes, because the torus has
degenerate eigenvalues and eigenvectors can rotate within a true eigenspace.

## Experiment Matrix

Run the manifold graph-mode experiment matrix:

```bash
uv run python grm_tcm_experiments.py --suite manifold --generate
```

This trains and evaluates:

```text
feature_only
feature_only_diffusion
feature_temporal_treatment
```

Output:

```text
grm_tcm_experiments/manifold_graph_mode_leaderboard.csv
```

The leaderboard puts prediction and geometry metrics side by side:

```text
next_day_grm_plus_lag_r2
flare_grm_plus_lag_auc
constitution_grm_mean_r2
saved_grm_lbo_mean_best_abs_corr
saved_grm_lbo_largest_subspace_mean_cos2
oracle_lbo_largest_subspace_mean_cos2
observation_lbo_largest_subspace_mean_cos2
```

Interpretation:

- `feature_temporal_treatment` is usually the prediction graph.
- `feature_only_diffusion` is the geometry-recovery graph.
- `oracle_torus_diffusion` is an upper-bound geometry control, not deployable.

## Prediction on New Visits

After static training, score new visits:

```bash
uv run python predict.py --visits NEW_VISITS.csv \
    --static-model grm_tcm_results/model \
    --dynamic-model grm_tcm_dynamic/model \
    --projection surrogate \
    --out predictions.csv
```

Inputs must include:

```text
subject_id
day
sleep_quality
hrv
resting_hr
body_temp
fatigue
pain
appetite
bowel_quality
mood_calm
energy
heaviness
cold_hot
```

Projection modes:

- `--projection surrogate`: persisted Ridge regressor from observations to saved embeddings.
- `--projection nystrom`: feature-only KNN/RBF extension of the spectral basis.

Neither projection perfectly reconstructs graph position for new visits when
the training graph used temporal/treatment edges. Treat projected `grm_mode_*`
as coordinates for the saved prediction heads, not as proof of exact graph
placement.

## Current Interpretation

The benchmark now separates two claims:

```text
Prediction claim:
  GRM features can improve future-state prediction over simple baselines.

Geometry claim:
  GRM graph modes recover a known Laplace-Beltrami basis.
```

These do not always move together. Temporal/treatment edges can help prediction
while making the operator less like a pure manifold Laplacian. The manifold
benchmark and experiment matrix are designed to expose that tradeoff directly.
