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
| `grm_tcm_dynamic_grm.py` | Rolling-window GRM diagnostics: G^(t), regime-change, self-resonance, transition reliability |
| `grm_tcm_persistence.py` | Manifest + npz/joblib helpers shared by train and dynamic pipelines |
| `grm_tcm_load.py` | `load_static_model` / `load_dynamic_model` — typed dataclass loaders |
| `predict.py` | Out-of-sample inference via surrogate (default) or Nyström extension; no retraining |
| `grm_tcm_dynamic_eval.py` | Falsifiable-verdicts eval: bootstrap CIs, JSON certificate, boxed-table console summary |
| `tests/` | pytest suite (`uv run pytest`) |
| `pyproject.toml` | Deps: numpy, pandas, scipy, scikit-learn, matplotlib, joblib, pyarrow (managed by uv) |

## Persisted model artifacts

Each output dir gets a `model/` subdirectory written next to the existing CSVs.

`grm_tcm_results/model/`:
- `manifest.json` — config, git sha, input hashes, package versions, schema `static-v2` (loader still accepts `static-v1`)
- `obs_preprocessor.joblib` — median-impute + standardize over the 12 observations
- `grm_basis.npz` — eigenvalues, eigenvectors, `rho`, `normalized` flag, `train_degrees`
- `nn_index.joblib` + `knn_sigma.json` — KNN attach for Nyström extension
- `ridge_next_day.joblib`, `logistic_flare.joblib` — prediction heads
- `embedding_surrogate.joblib` — Ridge `X_obs → embeddings` for predict.py's default projection mode (added in `static-v2`)
- `split_indices.json` — train/test indices used in evaluation
- `procrustes_R.npy` — synthetic-only latent-recovery rotation
- `visit_index.parquet` — `(visit_id, subject_id, day)` aligned with eigenvector rows

`grm_tcm_dynamic/model/`:
- `manifest.json` — schema `dynamic-v1`; `extra.static_manifest_sha` cross-links the static manifest. Loaders refuse mismatches.
- `state_preprocessor.joblib`, `state_kmeans.joblib`, `state_centroids.npy`, `state_metadata.json` — state vocabulary; KMeans is absent when `state_source="true_regime"`.
- `state_weights_visit.npy` — V × K soft RBF assignment matrix.
- `G_matrices.npz`, `grm_transition_matrices.npz`, `markov_transition_matrices.npz` — keyed by window end-day.
- `spectral_basis_per_window.npz` + `spectral_basis_sidecar.json` — Λ^(t), Ψ^(t), r_s, selected_modes per window.
- `window_index.parquet` — per-window audit log.

`predict.py` consumes both manifests and supports two projection modes:
- `--projection surrogate` (default, `static-v2`+): persisted Ridge `X_obs → embeddings`. Deterministic, fast. Note: the synthetic-benchmark GRM embeddings carry graph-position information that is not a function of observations alone, so the surrogate does not faithfully reproduce the spectral coordinates — it is a useful proxy for the downstream ridge/logistic heads, nothing more.
- `--projection nystrom`: feature-only KNN + RBF extension of the spectral basis. Approximate for the same reason (training graph has temporal + treatment + mutual-KNN edges that feature-only Nyström cannot reconstruct). Requires a graph_mode that includes KNN.

Both modes feed the ridge/logistic heads identically. Manifest schema is bumped to `static-v2`; the loader still accepts `static-v1` artifacts (surrogate will be absent, surrogate projection raises).

## Falsifiable-verdicts eval

`grm_tcm_dynamic_eval.py` consumes the persisted static + dynamic models and writes a structured `dynamic_eval_certificates.json` (bootstrap CIs per claim) plus a boxed Unicode summary table at end-of-run. Verdict row labels live in `grm_tcm_dynamic_eval.VERDICT_LABELS` (raw verdict key → display name); that dict also determines render order. Adding a new verdict: emit it from `write_certificate` and add a display label to the map. The renderer (`render_verdicts_table`) falls back to the raw key if no label is registered, so a missing label is visible but non-blocking. Box drawing is hand-rolled; no extra deps.

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

- Plots: matplotlib only, no seaborn, no custom colors. All `fig.savefig` calls funnel through `grm_tcm_plot_captions.save_with_caption(fig, path)`, which embeds a wording-from-registry caption strip below the axes. Captions live in `grm_tcm_plot_captions.CAPTIONS` keyed by `path.stem`. To add a plot: write the figure, call `save_with_caption`, add a 1–3 sentence entry to `CAPTIONS`. A missing entry triggers a runtime warning and falls back to a guardrail string. The `tests/test_plot_captions.py` registry-coverage test enforces every produced stem is registered.
- Random seed 42 throughout for reproducibility.
- Group splits use `subject_id` to avoid leakage.
- Type hints + short docstrings on public methods.
- Graceful degradation when optional inputs are missing.
