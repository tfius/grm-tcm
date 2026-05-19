# GRM-TCM Synthetic Benchmark

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Run

Generate the synthetic dataset, then train/evaluate the GRM model:

```bash
uv run python grm_tcm_synthetic_generator.py
uv run python grm_tcm_train.py
uv run python grm_tcm_diagnostics.py
uv run python grm_tcm_dynamic_grm.py
```

It generates:

```text
synthetic_grm_tcm/
  subjects.csv
  visits.csv
  latent_states.csv
  events.csv
  metadata.json

grm_tcm_results/
  grm_visit_embeddings.csv
  grm_feature_modes.csv
  grm_predictions.csv
  grm_metrics.json

grm_tcm_diagnostics/
  diagnostics_summary.json
  cluster_scores.csv
  contrarian_findings.csv
  regime_label_mismatch.csv
  plots/*.png

grm_tcm_dynamic/
  dynamic_grm_metrics.json
  state_source_comparison.csv
  state_assignments.csv
  rolling_regime_scores.csv
  spectral_energy.csv
  self_resonance_scores.csv
  subject_dynamic_scores.csv
  subject_resonance_summary.csv
  inferred_state_true_regime_confusion.csv
  grm_transition_predictions.csv
  subject_transition_predictions.csv
  transition_reliability.csv
```

Both the trainer and the dynamic pipeline also write a `model/` subdirectory containing the fitted preprocessor, eigenbasis, KMeans, regressors, G-matrices, and a `manifest.json` (config + git sha + input hashes + schema version). The dynamic manifest cross-references the static manifest sha, so a stale static model is rejected at load time.

## Predicting on new visits

After training, run `predict.py` to score new visits without retraining:

```bash
uv run python predict.py --visits NEW_VISITS.csv \
    --static-model grm_tcm_results/model \
    --dynamic-model grm_tcm_dynamic/model \
    --projection surrogate \
    --out predictions.csv
```

Inputs must include the 12 observation columns plus `subject_id` and `day`. Outputs include `grm_mode_*` coordinates, `pred_next_day_score`, `pred_flare_prob`, and (with `--dynamic-model`) `state_id`, `self_resonance`, `soft_self_resonance`, `top1_next_state`, `top1_next_state_prob`.

Two projection modes are supported:
- `--projection surrogate` (default): persisted Ridge regressor `X_obs → embeddings`. Deterministic; the recommended path for downstream prediction.
- `--projection nystrom`: feature-only KNN extension of the spectral basis. Available when the static model used a graph_mode that includes KNN.

Neither mode faithfully reproduces the GRM spectral embedding for new visits — those coordinates depend on multi-relational graph position (temporal + treatment + KNN edges), not on observations alone. Both modes are useful proxies for the downstream ridge/logistic heads; nothing more should be read into the `grm_mode_*` values in `predictions.csv`.

## Strict inductive evaluation

The default `grm_tcm_train.py` invocation is *transductive*: the visit graph and eigenbasis see every visit, then outcome metrics are reported on a within-graph train/test split — so the eigenbasis has already seen the "test" rows. To get honest held-out metrics, run:

```bash
uv run python grm_tcm_train.py --inductive \
    --projection surrogate \
    --output-dir grm_tcm_results_inductive
```

This splits subjects first (seed-controlled), fits scaler/NN-index/graph/eigenbasis/surrogate/heads on train subjects only, then projects test-subject visits via `--projection {surrogate, nystrom}` and scores them with the persisted heads. Results go to `inductive_eval_metrics.json` next to the standard CSVs; the persisted model in `model/` is the train-only fit and its `manifest.json` carries `extra.inductive: true` plus the held-out subject IDs for audit.

Compare against the transductive numbers in `grm_tcm_results/grm_metrics.json` to see how much of the apparent signal is graph-leak vs. real generalization.

## Tests

```bash
uv sync --group dev
uv run pytest
```

## Falsifiable-verdicts eval

After training the static + dynamic models, run the eval pipeline to score whether the headline GRM claims survive bootstrap CIs:

```bash
uv run python grm_tcm_dynamic_eval.py \
    --static-model-dir grm_tcm_results/model \
    --dynamic-model-dir grm_tcm_dynamic/model
```

It writes `dynamic_eval_certificates.json` and prints a boxed summary at end-of-run, e.g.:

```text
┌───────────────────────────────────────────┬────────┬──────────────────┬──────────┐
│                    Test                   │   Δ    │      95% CI      │ Verdict  │
├───────────────────────────────────────────┼────────┼──────────────────┼──────────┤
│ T1 GRM separates aliased states (entropy) │ +0.182 │ [+0.163, +0.201] │ PASS     │
│ T2 attractor AUC lift on aliased          │ +0.052 │ [+0.030, +0.075] │ PASS     │
│ ...                                       │ ...    │ ...              │ ...      │
└───────────────────────────────────────────┴────────┴──────────────────┴──────────┘
```

Row labels and order live in `grm_tcm_dynamic_eval.VERDICT_LABELS`. Verdicts present in the certificate but not in the map render under their raw key.

To run the multi-seed difficulty/ablation sweep:

```bash
uv run python grm_tcm_experiments.py
uv run python grm_tcm_experiments.py --state-sources kmeans_observation,kmeans_dynamic,true_regime
```

Inspect `grm_tcm_diagnostics/diagnostics_summary.json` first for a single run, or `grm_tcm_experiments/experiment_summary.json` after a sweep.

## Plots

Yes, the diagnostics scripts write plots.

Most useful static GRM plots:

- `grm_tcm_diagnostics/plots/grm_latent_correlation_heatmap.png` — whether GRM modes recover the known synthetic latent variables.
- `grm_tcm_diagnostics/plots/predicted_vs_actual_next_day_score.png` — whether outcome predictions are calibrated or just biased.
- `grm_tcm_diagnostics/plots/residual_histogram.png` — whether prediction errors are centered or systematically skewed.
- `grm_tcm_diagnostics/plots/grm_modes_scatter_hidden_subtype.png` — whether embeddings separate true hidden subtypes.
- `grm_tcm_diagnostics/plots/grm_modes_scatter_true_regime.png` — whether embeddings separate the simulator's true regimes.
- `grm_tcm_diagnostics/plots/grm_modes_scatter_tcm_like_label.png` — whether embeddings separate semantic TCM-like labels.
- `grm_tcm_diagnostics/plots/true_regime_occupancy_by_hidden_subtype.png` — whether the generator's hidden subtype changes regime occupancy.
- `grm_tcm_diagnostics/plots/true_regime_distribution_by_tcm_label.png` — how noisy TCM-like labels merge or split true regimes.
- `grm_tcm_diagnostics/plots/mean_grm_modes_by_hidden_subtype.png` and `mean_grm_modes_by_tcm_like_label.png` — which modes are associated with each grouping.

Most useful dynamic GRM plots:

- `grm_tcm_dynamic/plots/rolling_regime_change_score.png` — whether rolling `G^(t)` changes before flares or crashes.
- `grm_tcm_dynamic/plots/self_resonance_vs_dysregulation.png` — whether high self-resonance `G_ii` behaves like a stuck-state / attractor score.
- `grm_tcm_dynamic/plots/soft_self_resonance_vs_dysregulation.png` — the same attractor signal using soft visit-to-state assignment instead of hard state lookup.
- `grm_tcm_dynamic/plots/subject_regime_change_score.png` — whether individual subject-level `G_s^(t)` changes before subject-level events.
- `grm_tcm_dynamic/plots/subject_self_resonance_vs_dysregulation.png` — whether subject-conditioned self-resonance improves stuck-state interpretation.
- `grm_tcm_dynamic/plots/pooled_transition_reliability.png` and `subject_transition_reliability.png` — whether GRM transition probabilities are better calibrated than Markov-only probabilities.
- `grm_tcm_dynamic/plots/inferred_state_true_regime_confusion.png` — whether inferred state IDs separate the simulator's true regimes.
- `grm_tcm_dynamic/plots/true_stuck_occupancy_by_hidden_subtype.png` — whether hidden subtype drives attractor occupancy as intended.
- `grm_tcm_dynamic/plots/subject_resonance_vs_true_stuck_occupancy.png` — whether subject-level resonance tracks true stuck-regime occupancy.
- `grm_tcm_dynamic/plots/selected_modes_over_time.png` — whether the energy-selected number of modes is stable or changes across regimes.
- `grm_tcm_dynamic/plots/selected_modes_saturation.png` — whether selected modes are hitting the `max_modes` cap.
- `grm_tcm_dynamic/plots/cumulative_spectral_energy.png` — whether the spectrum changes across rolling windows even when selected mode count is flat.
- `grm_tcm_dynamic/plots/mean_cumulative_spectral_energy.png` — average energy captured by each mode rank.
- `grm_tcm_dynamic/plots/state_source_metric_comparison.png` — whether observation states, dynamic-feature states, or oracle true-regime states explain the gap.

The most meaningful visual comparison is:

1. Check `grm_latent_correlation_heatmap.png` for latent recovery.
2. Compare hidden-subtype vs TCM-label scatter plots to see whether GRM follows true hidden structure or semantic labels.
3. Check `rolling_regime_change_score.png` and `self_resonance_vs_dysregulation.png` to see whether the dynamic propagator adds signal beyond static embeddings.

If `selected_modes_over_time.png` is flat, inspect `selected_modes_saturation.png` and `cumulative_spectral_energy.png`. A flat selected-mode line often means the 95% energy rule is hitting `--max-modes`, not that the spectrum is truly constant.

Useful dynamic GRM options:

```bash
uv run python grm_tcm_dynamic_grm.py --similarity-mode knn --state-similarity-k 3
uv run python grm_tcm_dynamic_grm.py --similarity-mode threshold --similarity-quantile 0.8
uv run python grm_tcm_dynamic_grm.py --state-fit-end-day 40
uv run python grm_tcm_dynamic_grm.py --state-source kmeans_dynamic
uv run python grm_tcm_dynamic_grm.py --state-source true_regime
uv run python grm_tcm_dynamic_grm.py --compare-state-sources
```

The first two sharpen the state graph when the Laplacian spectrum is too flat. `--state-fit-end-day` fits state centroids only on early visits and assigns later visits into that fixed state space, which is closer to an out-of-sample diagnostic.

State sources control the discrete vocabulary used by dynamic GRM. `kmeans_observation` tests visible observation clusters, `kmeans_dynamic` adds short trajectory features to reduce aliasing, and `true_regime` is an oracle ceiling for the synthetic benchmark. `--compare-state-sources` writes `state_source_comparison.csv` and a comparison plot.

Dynamic GRM now reports hard and soft self-resonance. Soft self-resonance weights each visit by RBF similarity to all state centroids, which removes the discrete stripe artifact from hard `G_ii` lookup. `subject_resonance_summary.csv` aggregates soft self-resonance per subject so it can be compared with subject-level metadata such as `hidden_subtype`.

These plots are benchmark diagnostics only. They do not validate TCM, Qi, or a biological mechanism.
