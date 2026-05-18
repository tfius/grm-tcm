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
  plots/*.png

grm_tcm_dynamic/
  dynamic_grm_metrics.json
  rolling_regime_scores.csv
  spectral_energy.csv
  self_resonance_scores.csv
  subject_dynamic_scores.csv
  subject_resonance_summary.csv
  grm_transition_predictions.csv
  subject_transition_predictions.csv
  transition_reliability.csv
```

To run the multi-seed difficulty/ablation sweep:

```bash
uv run python grm_tcm_experiments.py
```

Inspect `grm_tcm_diagnostics/diagnostics_summary.json` first for a single run, or `grm_tcm_experiments/experiment_summary.json` after a sweep.

## Plots

Yes, the diagnostics scripts write plots.

Most useful static GRM plots:

- `grm_tcm_diagnostics/plots/grm_latent_correlation_heatmap.png` — whether GRM modes recover the known synthetic latent variables.
- `grm_tcm_diagnostics/plots/predicted_vs_actual_next_day_score.png` — whether outcome predictions are calibrated or just biased.
- `grm_tcm_diagnostics/plots/residual_histogram.png` — whether prediction errors are centered or systematically skewed.
- `grm_tcm_diagnostics/plots/grm_modes_scatter_hidden_subtype.png` — whether embeddings separate true hidden subtypes.
- `grm_tcm_diagnostics/plots/grm_modes_scatter_tcm_like_label.png` — whether embeddings separate semantic TCM-like labels.
- `grm_tcm_diagnostics/plots/mean_grm_modes_by_hidden_subtype.png` and `mean_grm_modes_by_tcm_like_label.png` — which modes are associated with each grouping.

Most useful dynamic GRM plots:

- `grm_tcm_dynamic/plots/rolling_regime_change_score.png` — whether rolling `G^(t)` changes before flares or crashes.
- `grm_tcm_dynamic/plots/self_resonance_vs_dysregulation.png` — whether high self-resonance `G_ii` behaves like a stuck-state / attractor score.
- `grm_tcm_dynamic/plots/soft_self_resonance_vs_dysregulation.png` — the same attractor signal using soft visit-to-state assignment instead of hard state lookup.
- `grm_tcm_dynamic/plots/subject_regime_change_score.png` — whether individual subject-level `G_s^(t)` changes before subject-level events.
- `grm_tcm_dynamic/plots/subject_self_resonance_vs_dysregulation.png` — whether subject-conditioned self-resonance improves stuck-state interpretation.
- `grm_tcm_dynamic/plots/pooled_transition_reliability.png` and `subject_transition_reliability.png` — whether GRM transition probabilities are better calibrated than Markov-only probabilities.
- `grm_tcm_dynamic/plots/selected_modes_over_time.png` — whether the energy-selected number of modes is stable or changes across regimes.
- `grm_tcm_dynamic/plots/selected_modes_saturation.png` — whether selected modes are hitting the `max_modes` cap.
- `grm_tcm_dynamic/plots/cumulative_spectral_energy.png` — whether the spectrum changes across rolling windows even when selected mode count is flat.
- `grm_tcm_dynamic/plots/mean_cumulative_spectral_energy.png` — average energy captured by each mode rank.

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
```

The first two sharpen the state graph when the Laplacian spectrum is too flat. `--state-fit-end-day` fits state centroids only on early visits and assigns later visits into that fixed state space, which is closer to an out-of-sample diagnostic.

Dynamic GRM now reports hard and soft self-resonance. Soft self-resonance weights each visit by RBF similarity to all state centroids, which removes the discrete stripe artifact from hard `G_ii` lookup. `subject_resonance_summary.csv` aggregates soft self-resonance per subject so it can be compared with subject-level metadata such as `hidden_subtype`.

These plots are benchmark diagnostics only. They do not validate TCM, Qi, or a biological mechanism.
