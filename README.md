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
```

I also tested the full run. It generates:

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
```

I adjusted defaults so it runs faster and avoids the earlier edge case where the flare target had only one class. The regenerated files follow the same synthetic generator and GRM trainer/evaluator design we outlined earlier.
