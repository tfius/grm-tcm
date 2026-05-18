# GRM-TCM Project Guide

Research prototype for **GRM-style spectral latent-state modeling** on a controlled synthetic longitudinal dataset. Not a biological simulator. Not a proof of TCM or Qi. Treat all outputs as benchmark diagnostics on a known generator.

## Pipeline (run in order)

```
python grm_tcm_synthetic_generator.py   # produces synthetic_grm_tcm/
python grm_tcm_train.py                 # produces grm_tcm_results/
python grm_tcm_diagnostics.py           # produces grm_tcm_diagnostics/
```

Each stage reads only the outputs of the previous stages, so they are independently re-runnable.

## Layout

| File / Dir | Role |
|---|---|
| `grm_tcm_synthetic_generator.py` | Generator: simulates latent dynamics + observations + naive Qi/TCM-like labels with intentional ontology mismatch |
| `grm_tcm_train.py` | Trainer: builds visit-state graph, Laplacian eigenmodes, GRM embeddings, predicts `next_day_score` / `flare_next_day`, compares to raw + naive baselines |
| `grm_tcm_diagnostics.py` | Diagnostics: latent recovery, ontology mismatch, clustering vs labels, prediction error analysis, plots |
| `synthetic_grm_tcm/` | Generator outputs (`subjects.csv`, `visits.csv`, `latent_states.csv`, `events.csv`, `metadata.json`) |
| `grm_tcm_results/` | Trainer outputs (`grm_visit_embeddings.csv`, `grm_feature_modes.csv`, `grm_predictions.csv`, `grm_metrics.json`) |
| `grm_tcm_diagnostics/` | Diagnostics outputs (CSV tables, `diagnostics_summary.json`, `plots/*.png`) |
| `pyproject.toml` | Deps: numpy, pandas, scipy, scikit-learn, matplotlib (managed by uv) |

## Key data schemas

- **visits.csv** join key: `(subject_id, day)`. Contains 12 observations, derived `global_dysregulation_score`, next-day targets (`next_day_score`, `flare_next_day`, `crash_next_day`, `worsening_2day`), true latents merged in, and semantic labels (`qi_like_label`, `tcm_like_label`, `contrarian_signature`).
- **latent_states.csv** true latents: `vitality_depletion`, `stress_activation`, `inflammatory_load`, `digestive_instability`.
- **subjects.csv** carries `hidden_subtype` (0/1/2) — the *real* generative cluster, deliberately not aligned with `tcm_like_label`.
- **grm_visit_embeddings.csv** GRM modes `grm_mode_1..N` keyed by `visit_id` / `(subject_id, day)`.

## Scientific framing (important)

Always describe results in these terms:
- **latent-state recovery** — does GRM align with the true generated latents?
- **ontology mismatch detection** — do naive labels split or merge against hidden subtypes?
- **synthetic benchmark validation** — known generator, so we can score recovery quantitatively.
- **possible contrarian pattern discovery** — labels that are impure relative to the real structure.

Do **not** claim this validates TCM, Qi, or any clinical ontology. The labels in the dataset are intentionally mislabeled in places (`split_spleen_like_label`, `merge_liver_damp_like_labels`) so the diagnostics can find that mismatch.

## Conventions

- Plots: matplotlib only, no seaborn, no custom colors.
- Random seed 42 throughout for reproducibility.
- Group splits use `subject_id` to avoid leakage.
- Type hints + short docstrings on public methods.
- Graceful degradation when optional inputs are missing.
