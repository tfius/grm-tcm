from __future__ import annotations

"""
Loaders for persisted static and dynamic GRM-TCM models.

Provides dataclass bundles that hydrate every artifact written by
GRMTCMTrainer._save_model() and save_dynamic_model(). The loaders verify
schema versions and (for dynamic) validate the static cross-link sha so a
re-train invalidates downstream models loudly rather than silently.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from grm_tcm_persistence import (
    load_int_keyed_npz,
    load_joblib,
    manifest_sha,
    read_manifest,
)


STATIC_SCHEMA_VERSIONS = ["static-v1", "static-v2"]
DYNAMIC_SCHEMA_VERSIONS = ["dynamic-v1"]


@dataclass
class StaticGRMModel:
    """Hydrated static GRM-TCM model with everything needed for Nyström inference."""

    manifest: Dict[str, Any]
    config: Dict[str, Any]
    obs_preprocessor: Any
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    rho: float
    normalized: bool
    n_modes: int
    graph_mode: str
    train_degrees: Optional[np.ndarray] = None
    nn_index: Optional[Any] = None
    knn_sigma: Optional[float] = None
    ridge_reg: Optional[Any] = None
    logistic_clf: Optional[Any] = None
    flare_temperature: Optional[float] = None
    embedding_surrogate: Optional[Any] = None
    procrustes_R: Optional[np.ndarray] = None
    visit_index: Optional[pd.DataFrame] = None
    split_indices: Optional[Dict[str, Any]] = None
    feature_names: Optional[List[str]] = None


@dataclass
class DynamicGRMModel:
    """Hydrated dynamic GRM model with G^(t), transitions, spectral basis, state model."""

    manifest: Dict[str, Any]
    config: Dict[str, Any]
    state_source: str
    soft_sigma: float
    feature_columns: List[str]
    state_preprocessor: Optional[Any]
    state_kmeans: Optional[Any]
    state_centroids: np.ndarray
    state_weights: np.ndarray
    global_g_matrices: Dict[int, np.ndarray]
    grm_transition_matrices: Dict[int, np.ndarray]
    markov_transition_matrices: Dict[int, np.ndarray]
    spectral_basis_per_window: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    window_index: Optional[pd.DataFrame] = None


def load_static_model(model_dir: Path) -> StaticGRMModel:
    """Hydrate a static GRM-TCM model from disk."""

    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir, allowed_schema_versions=STATIC_SCHEMA_VERSIONS)
    basis = np.load(model_dir / "grm_basis.npz")
    obs_preprocessor = load_joblib(model_dir / "obs_preprocessor.joblib")

    nn_index = None
    knn_sigma = None
    nn_path = model_dir / "nn_index.joblib"
    if nn_path.exists():
        nn_index = load_joblib(nn_path)
        sigma_path = model_dir / "knn_sigma.json"
        if sigma_path.exists():
            knn_sigma = json.load(open(sigma_path))["knn_sigma"]

    ridge_reg = load_joblib(model_dir / "ridge_next_day.joblib") if (model_dir / "ridge_next_day.joblib").exists() else None
    logistic_clf = load_joblib(model_dir / "logistic_flare.joblib") if (model_dir / "logistic_flare.joblib").exists() else None
    embedding_surrogate = (
        load_joblib(model_dir / "embedding_surrogate.joblib")
        if (model_dir / "embedding_surrogate.joblib").exists()
        else None
    )
    flare_temperature: Optional[float] = None
    flare_temp_path = model_dir / "flare_temperature.json"
    if flare_temp_path.exists():
        with open(flare_temp_path, "r", encoding="utf-8") as f:
            flare_temperature = float(json.load(f).get("T", 1.0))

    procrustes_R = np.load(model_dir / "procrustes_R.npy") if (model_dir / "procrustes_R.npy").exists() else None
    visit_index = pd.read_parquet(model_dir / "visit_index.parquet") if (model_dir / "visit_index.parquet").exists() else None
    split = json.load(open(model_dir / "split_indices.json")) if (model_dir / "split_indices.json").exists() else None

    feature_names = (manifest.get("extra") or {}).get("feature_names")
    if feature_names is not None:
        feature_names = list(feature_names)

    return StaticGRMModel(
        manifest=manifest,
        config=manifest["config"],
        obs_preprocessor=obs_preprocessor,
        eigenvalues=basis["eigenvalues"].astype(float),
        eigenvectors=basis["eigenvectors"].astype(float),
        rho=float(basis["rho"]),
        normalized=bool(basis["normalized"]),
        n_modes=int(basis["n_modes"]),
        graph_mode=str(basis["graph_mode"]),
        train_degrees=basis["train_degrees"].astype(float) if "train_degrees" in basis.files else None,
        nn_index=nn_index,
        knn_sigma=knn_sigma,
        ridge_reg=ridge_reg,
        logistic_clf=logistic_clf,
        flare_temperature=flare_temperature,
        embedding_surrogate=embedding_surrogate,
        procrustes_R=procrustes_R,
        visit_index=visit_index,
        split_indices=split,
        feature_names=feature_names,
    )


def load_dynamic_model(
    model_dir: Path,
    *,
    static_model: Optional[StaticGRMModel] = None,
    static_model_dir: Optional[Path] = None,
) -> DynamicGRMModel:
    """Hydrate a dynamic GRM model. If a static model is supplied, validate the cross-link sha."""

    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir, allowed_schema_versions=DYNAMIC_SCHEMA_VERSIONS)

    expected_sha = (manifest.get("extra") or {}).get("static_manifest_sha")
    if static_model_dir is not None and expected_sha is not None:
        actual = manifest_sha(static_model_dir)
        if actual != expected_sha:
            raise ValueError(
                f"Dynamic manifest cross-link static_manifest_sha {expected_sha!r} does not match "
                f"current static model at {static_model_dir} (sha={actual!r}). Re-run dynamic pipeline "
                f"or supply the matching static model."
            )

    state_metadata = json.load(open(model_dir / "state_metadata.json"))
    state_preprocessor = (
        load_joblib(model_dir / "state_preprocessor.joblib") if (model_dir / "state_preprocessor.joblib").exists() else None
    )
    state_kmeans = (
        load_joblib(model_dir / "state_kmeans.joblib") if (model_dir / "state_kmeans.joblib").exists() else None
    )
    state_centroids = np.load(model_dir / "state_centroids.npy")
    state_weights = np.load(model_dir / "state_weights_visit.npy")

    global_g = load_int_keyed_npz(model_dir / "G_matrices.npz", prefix="d")
    grm_trans = load_int_keyed_npz(model_dir / "grm_transition_matrices.npz", prefix="d")
    markov_trans = load_int_keyed_npz(model_dir / "markov_transition_matrices.npz", prefix="d")

    basis: Dict[int, Dict[str, Any]] = {}
    basis_path = model_dir / "spectral_basis_per_window.npz"
    sidecar_path = model_dir / "spectral_basis_sidecar.json"
    if basis_path.exists() and sidecar_path.exists():
        sidecar = json.load(open(sidecar_path))
        with np.load(basis_path) as f:
            for end_day_str, scalars in sidecar.items():
                end_day = int(end_day_str)
                lambdas_key = f"lambdas_d{end_day}"
                psi_key = f"psi_d{end_day}"
                if lambdas_key not in f.files or psi_key not in f.files:
                    continue
                basis[end_day] = {
                    "lambdas": f[lambdas_key].copy(),
                    "psi": f[psi_key].copy(),
                    "r_s": float(scalars["r_s"]),
                    "selected_modes": int(scalars["selected_modes"]),
                }

    window_index = (
        pd.read_parquet(model_dir / "window_index.parquet")
        if (model_dir / "window_index.parquet").exists()
        else None
    )

    return DynamicGRMModel(
        manifest=manifest,
        config=manifest["config"],
        state_source=state_metadata["source"],
        soft_sigma=float(state_metadata["soft_sigma"]),
        feature_columns=list(state_metadata["feature_columns"]),
        state_preprocessor=state_preprocessor,
        state_kmeans=state_kmeans,
        state_centroids=state_centroids,
        state_weights=state_weights,
        global_g_matrices=global_g,
        grm_transition_matrices=grm_trans,
        markov_transition_matrices=markov_trans,
        spectral_basis_per_window=basis,
        window_index=window_index,
    )
