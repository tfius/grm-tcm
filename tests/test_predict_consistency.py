"""Smoke + sanity test for predict.py end-to-end on held-out visits."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grm_tcm_dynamic_grm import DynamicGRMConfig, run_dynamic
from grm_tcm_load import load_static_model
from grm_tcm_train import GRMTCMTrainer, GRMTrainConfig
from predict import (
    OBSERVATION_NAMES,
    apply_dynamic_scores,
    apply_static_heads,
    nystrom_grm_coordinates,
    surrogate_grm_coordinates,
)
from grm_tcm_load import load_dynamic_model


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    static_dir = tmp_path_factory.mktemp("grm_static")
    dyn_dir = tmp_path_factory.mktemp("grm_dyn")
    GRMTCMTrainer(
        GRMTrainConfig(
            input_dir="synthetic_grm_tcm",
            output_dir=str(static_dir),
            graph_mode="feature_temporal_treatment",
            n_modes=4,
        )
    ).run()
    run_dynamic(
        DynamicGRMConfig(
            data_dir="synthetic_grm_tcm",
            results_dir=str(static_dir),
            output_dir=str(dyn_dir),
            n_states=8,
            window_size=14,
            max_modes=4,
        )
    )
    return static_dir / "model", dyn_dir / "model"


def test_nystrom_produces_finite_coords(fitted):
    static_dir, _ = fitted
    static = load_static_model(static_dir)
    visits = pd.read_csv("synthetic_grm_tcm/visits.csv").sort_values(["subject_id", "day"]).reset_index(drop=True)
    last_days = visits["day"].max()
    held = visits[visits["day"] >= last_days - 4].head(50)
    X = static.obs_preprocessor.transform(held[OBSERVATION_NAMES].to_numpy(float))
    coords = nystrom_grm_coordinates(static, X, n_neighbors=12)
    assert coords.shape == (len(held), static.eigenvectors.shape[1])
    assert np.isfinite(coords).all()


def test_predict_pipeline_end_to_end(fitted):
    static_dir, dyn_dir = fitted
    static = load_static_model(static_dir)
    dynamic = load_dynamic_model(dyn_dir, static_model_dir=static_dir)
    visits = pd.read_csv("synthetic_grm_tcm/visits.csv").sort_values(["subject_id", "day"]).reset_index(drop=True)
    last_days = visits["day"].max()
    held = visits[visits["day"] >= last_days - 4].head(50)
    X = static.obs_preprocessor.transform(held[OBSERVATION_NAMES].to_numpy(float))
    coords = nystrom_grm_coordinates(static, X, n_neighbors=12)
    heads = apply_static_heads(static, coords)
    assert "pred_next_day_score" in heads and len(heads["pred_next_day_score"]) == len(held)
    assert "pred_flare_prob" in heads
    assert np.isfinite(heads["pred_next_day_score"]).all()
    assert ((heads["pred_flare_prob"] >= 0) & (heads["pred_flare_prob"] <= 1)).all()
    dyn_df = apply_dynamic_scores(dynamic, held.reset_index(drop=True))
    assert len(dyn_df) == len(held)
    assert {"state_id", "self_resonance", "soft_self_resonance", "top1_next_state_prob"}.issubset(dyn_df.columns)


def test_surrogate_projection_produces_finite_predictions(fitted):
    """Surrogate path returns finite coords + valid head outputs.

    Note: the surrogate is NOT expected to reproduce the saved spectral embeddings
    closely — GRM coordinates depend on multi-relational graph position, not just
    observations. The surrogate is a useful proxy for the downstream heads only.
    """

    static_dir, _ = fitted
    static = load_static_model(static_dir)
    assert static.embedding_surrogate is not None, "static-v2 model must persist embedding_surrogate"
    visits = pd.read_csv("synthetic_grm_tcm/visits.csv").sort_values(["subject_id", "day"]).reset_index(drop=True)
    last_days = visits["day"].max()
    held = visits[visits["day"] >= last_days - 4].head(50)
    X = static.obs_preprocessor.transform(held[OBSERVATION_NAMES].to_numpy(float))
    coords = surrogate_grm_coordinates(static, X)
    assert coords.shape == (len(held), static.eigenvectors.shape[1])
    assert np.isfinite(coords).all()
    heads = apply_static_heads(static, coords)
    assert np.isfinite(heads["pred_next_day_score"]).all()
    assert ((heads["pred_flare_prob"] >= 0) & (heads["pred_flare_prob"] <= 1)).all()


def test_surrogate_raises_when_absent(tmp_path):
    """Loading a (manually constructed) static-v1 model should leave surrogate=None
    and surrogate_grm_coordinates should raise a clear error."""

    static_dir, _ = _fresh_v1_model(tmp_path)
    static = load_static_model(static_dir)
    assert static.embedding_surrogate is None
    X = np.zeros((3, len(OBSERVATION_NAMES)))
    X = static.obs_preprocessor.transform(X)
    with pytest.raises(RuntimeError, match="embedding_surrogate"):
        surrogate_grm_coordinates(static, X)


def _fresh_v1_model(tmp_path: Path):
    """Train a model, then downgrade its manifest to static-v1 and delete the
    embedding_surrogate file to simulate a pre-v2 artifact."""

    import json

    static_dir = tmp_path / "v1_static"
    GRMTCMTrainer(
        GRMTrainConfig(
            input_dir="synthetic_grm_tcm",
            output_dir=str(static_dir),
            graph_mode="feature_temporal_treatment",
            n_modes=4,
        )
    ).run()
    model_dir = static_dir / "model"
    surrogate_path = model_dir / "embedding_surrogate.joblib"
    if surrogate_path.exists():
        surrogate_path.unlink()
    manifest = json.load(open(model_dir / "manifest.json"))
    manifest["schema_version"] = "static-v1"
    with open(model_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
    return model_dir, None
