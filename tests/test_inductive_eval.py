"""Tests for the strict inductive evaluation mode.

Verifies that subjects are genuinely disjoint, that the persisted model carries
the inductive manifest extras, and that both projection methods run end-to-end
and produce non-NaN held-out metrics.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grm_tcm_load import load_static_model
from grm_tcm_train import GRMTCMTrainer, GRMTrainConfig


@pytest.fixture(scope="module")
def inductive_surrogate_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("grm_inductive_surrogate")
    GRMTCMTrainer(
        GRMTrainConfig(
            input_dir="synthetic_grm_tcm",
            output_dir=str(out),
            graph_mode="feature_temporal_treatment",
            n_modes=4,
            inductive=True,
            projection="surrogate",
            test_size=0.25,
        )
    ).run()
    return out


@pytest.fixture(scope="module")
def inductive_nystrom_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("grm_inductive_nystrom")
    GRMTCMTrainer(
        GRMTrainConfig(
            input_dir="synthetic_grm_tcm",
            output_dir=str(out),
            graph_mode="feature_temporal_treatment",
            n_modes=4,
            inductive=True,
            projection="nystrom",
            test_size=0.25,
        )
    ).run()
    return out


def test_inductive_metrics_file_exists(inductive_surrogate_dir: Path):
    assert (inductive_surrogate_dir / "inductive_eval_metrics.json").exists()


def test_inductive_metrics_shape(inductive_surrogate_dir: Path):
    m = json.load(open(inductive_surrogate_dir / "inductive_eval_metrics.json"))
    assert m["evaluation_mode"] == "inductive"
    assert m["projection"] == "surrogate"
    assert m["n_train_subjects"] > 0
    assert m["n_test_subjects"] > 0
    for key in ["grm_ridge", "raw_random_forest", "naive_current_score"]:
        assert key in m["regression"]
        assert np.isfinite(m["regression"][key]["r2"])
    assert "grm_logistic" in m["classification"]


def test_subjects_are_disjoint(inductive_surrogate_dir: Path):
    """Train embeddings and test embeddings must come from disjoint subject sets."""

    df = pd.read_csv(inductive_surrogate_dir / "grm_visit_embeddings.csv")
    assert "split" in df.columns
    train_subjects = set(df[df["split"] == "train"]["subject_id"].unique())
    test_subjects = set(df[df["split"] == "test"]["subject_id"].unique())
    assert train_subjects.isdisjoint(test_subjects), \
        f"Train/test subject overlap: {train_subjects & test_subjects}"
    assert len(train_subjects) > 0 and len(test_subjects) > 0


def test_manifest_carries_inductive_extras(inductive_surrogate_dir: Path):
    m = json.load(open(inductive_surrogate_dir / "model" / "manifest.json"))
    extra = m.get("extra", {})
    assert extra.get("inductive") is True
    assert extra.get("projection") == "surrogate"
    assert extra.get("n_test_subjects", 0) > 0
    assert isinstance(extra.get("test_subjects"), list)


def test_persisted_model_loads(inductive_surrogate_dir: Path):
    model = load_static_model(inductive_surrogate_dir / "model")
    # Eigenvectors must be sized by train-subject visits only
    n_train_visits = int(json.load(open(inductive_surrogate_dir / "inductive_eval_metrics.json"))["n_train_visits"])
    assert model.eigenvectors.shape[0] == n_train_visits
    assert model.embedding_surrogate is not None


def test_nystrom_projection_runs(inductive_nystrom_dir: Path):
    m = json.load(open(inductive_nystrom_dir / "inductive_eval_metrics.json"))
    assert m["projection"] == "nystrom"
    assert np.isfinite(m["regression"]["grm_ridge"]["r2"])


def test_predictions_csv_is_test_only(inductive_surrogate_dir: Path):
    pred = pd.read_csv(inductive_surrogate_dir / "grm_predictions.csv")
    assert (pred["split"] == "test").all()
    assert "pred_grm_next_score" in pred.columns
    assert "pred_grm_flare_prob" in pred.columns
    assert pred["pred_grm_next_score"].notna().all()


def test_out_of_sample_latent_recovery_present(inductive_surrogate_dir: Path):
    m = json.load(open(inductive_surrogate_dir / "inductive_eval_metrics.json"))
    rec = m["latent_recovery"]
    assert "in_sample_train" in rec
    assert "out_of_sample_test" in rec
    assert np.isfinite(rec["out_of_sample_test"]["mean_abs_aligned_correlation"])


def test_constitution_recovery_reports_subject_graph(inductive_surrogate_dir: Path):
    m = json.load(open(inductive_surrogate_dir / "inductive_eval_metrics.json"))
    const = m.get("constitution_recovery", {})
    if not const:
        pytest.skip("synthetic subjects.csv does not include constitution targets")
    assert "subject_graph_grm_ridge" in const
    assert "subject_graph_grm_ridge" in const["mean_r2"]
    assert np.isfinite(const["mean_r2"]["subject_graph_grm_ridge"])


def test_pang_style_controls_reported(inductive_surrogate_dir: Path):
    m = json.load(open(inductive_surrogate_dir / "inductive_eval_metrics.json"))
    assert "smooth_rbf_kernel_ridge" in m["regression"]
    assert "smooth_rbf_kernel_ridge" in m["classification"]
    assert np.isfinite(m["regression"]["smooth_rbf_kernel_ridge"]["r2"])
    assert "parsimony" in m
    assert "grm_structural_hyperparameters" in m["parsimony"]
    concentration = m.get("spectral_signal_concentration", {})
    assert "next_day_score" in concentration
    assert "modes_for_75pct_signal" in concentration["next_day_score"]
