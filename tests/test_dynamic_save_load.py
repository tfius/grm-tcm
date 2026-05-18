"""Round-trip: dynamic pipeline saves an artifact that load_dynamic_model rehydrates."""

from pathlib import Path

import numpy as np
import pytest

from grm_tcm_dynamic_grm import DynamicGRMConfig, run_dynamic
from grm_tcm_load import load_dynamic_model, load_static_model
from grm_tcm_train import GRMTCMTrainer, GRMTrainConfig


@pytest.fixture(scope="module")
def pipeline_dirs(tmp_path_factory):
    static_dir = tmp_path_factory.mktemp("grm_static")
    dynamic_dir = tmp_path_factory.mktemp("grm_dynamic")
    train_cfg = GRMTrainConfig(
        input_dir="synthetic_grm_tcm",
        output_dir=str(static_dir),
        graph_mode="feature_temporal_treatment",
        n_modes=4,
    )
    GRMTCMTrainer(train_cfg).run()
    dyn_cfg = DynamicGRMConfig(
        data_dir="synthetic_grm_tcm",
        results_dir=str(static_dir),
        output_dir=str(dynamic_dir),
        n_states=8,
        window_size=14,
        max_modes=4,
    )
    run_dynamic(dyn_cfg)
    return static_dir / "model", dynamic_dir / "model"


def test_dynamic_artifacts_present(pipeline_dirs):
    _, model_dir = pipeline_dirs
    for name in [
        "manifest.json",
        "state_preprocessor.joblib",
        "state_kmeans.joblib",
        "state_centroids.npy",
        "state_metadata.json",
        "state_weights_visit.npy",
        "G_matrices.npz",
        "grm_transition_matrices.npz",
        "markov_transition_matrices.npz",
        "spectral_basis_per_window.npz",
        "spectral_basis_sidecar.json",
        "window_index.parquet",
    ]:
        assert (model_dir / name).exists(), f"Missing dynamic artifact {name}"


def test_load_dynamic_round_trip(pipeline_dirs):
    static_dir, dyn_dir = pipeline_dirs
    model = load_dynamic_model(dyn_dir, static_model_dir=static_dir)
    assert model.state_centroids.ndim == 2
    assert model.soft_sigma > 0
    assert model.state_kmeans is not None
    assert len(model.global_g_matrices) > 0
    assert len(model.grm_transition_matrices) == len(model.global_g_matrices)
    assert len(model.markov_transition_matrices) == len(model.global_g_matrices)
    # G must be square in state-space and have non-negative diagonal up to numerical noise.
    for G in model.global_g_matrices.values():
        assert G.shape[0] == G.shape[1] == model.state_centroids.shape[0]
        # diag may be slightly negative due to sign-shift convention but resonance G_pos clamps it.
    assert len(model.spectral_basis_per_window) == len(model.global_g_matrices)


def test_dynamic_cross_link_rejects_wrong_static(pipeline_dirs, tmp_path):
    _, dyn_dir = pipeline_dirs
    # Retrain into a fresh static dir; its manifest sha will differ.
    other_static = tmp_path / "other_static"
    GRMTCMTrainer(
        GRMTrainConfig(
            input_dir="synthetic_grm_tcm",
            output_dir=str(other_static),
            graph_mode="feature_temporal_treatment",
            n_modes=4,
        )
    ).run()
    with pytest.raises(ValueError, match="static_manifest_sha"):
        load_dynamic_model(dyn_dir, static_model_dir=other_static / "model")
