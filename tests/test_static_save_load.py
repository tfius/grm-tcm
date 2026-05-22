"""Round-trip: static trainer saves an artifact that load_static_model rehydrates."""

from pathlib import Path

import numpy as np
import pytest

from grm_tcm_load import load_static_model
from grm_tcm_train import GRMTCMTrainer, GRMTrainConfig


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("grm_static")
    cfg = GRMTrainConfig(
        input_dir="synthetic_grm_tcm",
        output_dir=str(out),
        graph_mode="feature_temporal_treatment",
        n_modes=4,
    )
    GRMTCMTrainer(cfg).run()
    return out / "model"


def test_artifacts_present(trained_model_dir: Path):
    for name in [
        "manifest.json",
        "obs_preprocessor.joblib",
        "grm_basis.npz",
        "nn_index.joblib",
        "knn_sigma.json",
        "ridge_next_day.joblib",
        "logistic_flare.joblib",
        "embedding_surrogate.joblib",
        "flare_temperature.json",
        "split_indices.json",
        "visit_index.parquet",
    ]:
        assert (trained_model_dir / name).exists(), f"Missing artifact {name}"


def test_flare_temperature_artifact_shape(trained_model_dir: Path):
    """flare_temperature.json must contain a positive T and the expected metadata fields."""

    import json
    payload = json.load(open(trained_model_dir / "flare_temperature.json"))
    assert "T" in payload and payload["T"] > 0
    assert payload.get("fit_method") == "inner_5fold_cv_on_train"
    assert payload.get("applies_to") == "logistic_flare"


def test_flare_temperature_loaded(trained_model_dir: Path):
    """Loader must expose flare_temperature as a float when the artifact is present."""

    model = load_static_model(trained_model_dir)
    assert model.flare_temperature is not None
    assert np.isfinite(model.flare_temperature) and model.flare_temperature > 0


def test_manifest_is_static_v2(trained_model_dir: Path):
    import json
    m = json.load(open(trained_model_dir / "manifest.json"))
    assert m["schema_version"] == "static-v3"
    assert m["extra"]["embedding_convention"] == "sqrt_grm_kernel_feature"


def test_load_static_round_trip(trained_model_dir: Path):
    model = load_static_model(trained_model_dir)
    assert model.eigenvectors.ndim == 2
    assert model.eigenvalues.shape[0] == model.eigenvectors.shape[1]
    assert model.rho > 0
    assert model.normalized is True
    assert model.knn_sigma is not None and model.knn_sigma > 0
    assert model.train_degrees is not None
    assert model.train_degrees.shape[0] == model.eigenvectors.shape[0]
    assert model.ridge_reg is not None
    assert model.logistic_clf is not None
    assert model.visit_index is not None and len(model.visit_index) == model.eigenvectors.shape[0]
    assert model.split_indices is not None
    assert {"seed", "test_size", "train", "test"}.issubset(model.split_indices.keys())


def test_eigenvectors_canonical_signs(trained_model_dir: Path):
    """First nonzero entry of each mode should be positive after canonicalization."""

    model = load_static_model(trained_model_dir)
    for j in range(model.eigenvectors.shape[1]):
        col = model.eigenvectors[:, j]
        first = col[np.argmax(np.abs(col) > 1e-12)]
        assert first > 0
