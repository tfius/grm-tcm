from __future__ import annotations

"""
Out-of-sample inference for persisted GRM-TCM models.

Usage:
    uv run python predict.py --visits NEW.csv \\
        --static-model grm_tcm_results/model \\
        [--dynamic-model grm_tcm_dynamic/model] \\
        --out predictions.csv

Pipeline:
  1. Load static GRM model (preprocessor, eigenbasis, regressors, optional surrogate).
  2. Apply obs_preprocessor to the new visits' observation columns.
  3. Project to GRM coordinates via one of two modes:
     - surrogate (default): persisted Ridge X_obs -> embeddings. Faithful and fast.
       Recovers training embeddings to high accuracy by construction.
     - nystrom: feature-only KNN + RBF extension of the spectral basis. Approximate
       because the training graph has multi-relational edges (temporal, treatment,
       mutual-KNN augmentation) that feature-only Nyström cannot reconstruct.
       Available when the static model was trained with a graph_mode that includes
       KNN (feature_only / feature_temporal / feature_temporal_treatment).
  4. Apply ridge/logistic heads for next_day_score / flare probability.
  5. (Optional) Apply state preprocessor + KMeans from the dynamic model, look up the
     most-recent G^(t) <= visit day, emit self-resonance / soft-self-resonance and
     top-1 next-state transition probability.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from grm_tcm_load import (
    DynamicGRMModel,
    StaticGRMModel,
    load_dynamic_model,
    load_static_model,
)
from grm_tcm_projection import nystrom_extend_arrays, surrogate_project


OBSERVATION_NAMES = [
    "sleep_quality", "hrv", "resting_hr", "body_temp", "fatigue", "pain",
    "appetite", "bowel_quality", "mood_calm", "energy", "heaviness", "cold_hot",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run GRM-TCM inference on new visits.")
    parser.add_argument("--visits", required=True, help="CSV with new visits (must contain observation columns + subject_id, day).")
    parser.add_argument("--static-model", default="grm_tcm_results/model", help="Path to static model dir.")
    parser.add_argument("--dynamic-model", default=None, help="Optional path to dynamic model dir.")
    parser.add_argument("--out", default="predictions.csv", help="Output CSV path.")
    parser.add_argument(
        "--projection",
        choices=["surrogate", "nystrom"],
        default="surrogate",
        help="How to project new visits into GRM coordinates. 'surrogate' uses the persisted Ridge head "
        "X_obs -> embeddings (default, recovers training visits exactly by construction). 'nystrom' uses "
        "feature-only KNN + RBF extension of the spectral basis (approximate; requires graph_mode with KNN).",
    )
    parser.add_argument("--n-neighbors", type=int, default=12, help="Neighbors used in Nyström extension.")
    return parser.parse_args()


def nystrom_grm_coordinates(
    static: StaticGRMModel,
    X_new_scaled: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Project new visits into the persisted GRM basis via Nyström extension.

    Returns coordinates of shape (m, n_modes), already weighted by 1/(1 + rho^2 * lambda).
    Thin wrapper around grm_tcm_projection.nystrom_extend_arrays.
    """

    if static.nn_index is None:
        raise RuntimeError(
            "Loaded static model has no KNN index. Nyström extension requires graph_mode "
            "including feature_only / feature_temporal / feature_temporal_treatment."
        )
    if static.knn_sigma is None:
        raise RuntimeError("Loaded static model is missing knn_sigma; cannot recompute RBF weights.")

    return nystrom_extend_arrays(
        X_new_scaled,
        nn_index=static.nn_index,
        knn_sigma=float(static.knn_sigma),
        eigenvalues=static.eigenvalues,
        eigenvectors=static.eigenvectors,
        rho=float(static.rho),
        normalized=bool(static.normalized),
        train_degrees=static.train_degrees,
        n_neighbors=int(n_neighbors),
    )


def surrogate_grm_coordinates(static: StaticGRMModel, X_new_scaled: np.ndarray) -> np.ndarray:
    """Project new visits via the persisted Ridge surrogate.

    Important caveat: the surrogate does NOT faithfully reproduce the spectral
    GRM embedding even on training visits. GRM coordinates depend on
    multi-relational graph position (KNN + mutual augmentation + temporal +
    treatment edges), not on observations alone, so no linear (or even kNN)
    model in X_obs can recover them with high fidelity. The surrogate is a
    deterministic, fast proxy for feeding the downstream ridge/logistic heads;
    its outputs should not be interpreted as 'the GRM coordinates for this
    visit.' See CLAUDE.md for the full caveat.
    """

    if static.embedding_surrogate is None:
        raise RuntimeError(
            "Loaded static model has no embedding_surrogate. The model was likely trained before "
            "static-v2 schema; retrain or use --projection nystrom."
        )
    return surrogate_project(static.embedding_surrogate, X_new_scaled)


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def apply_static_heads(static: StaticGRMModel, grm_coords: np.ndarray) -> Dict[str, np.ndarray]:
    """Run the persisted ridge/logistic heads against new GRM coordinates.

    If a fitted flare_temperature is present on the model, the calibrated flare
    probability is also emitted as `pred_flare_prob_calibrated`. Uncalibrated is
    kept for audit; downstream consumers should prefer the calibrated value
    when available.
    """

    out: Dict[str, np.ndarray] = {}
    if static.ridge_reg is not None:
        out["pred_next_day_score"] = static.ridge_reg.predict(grm_coords)
    if static.logistic_clf is not None:
        out["pred_flare_prob"] = static.logistic_clf.predict_proba(grm_coords)[:, 1]
        T = static.flare_temperature
        if T is not None and np.isfinite(T) and T > 0:
            z = static.logistic_clf.decision_function(grm_coords)
            out["pred_flare_prob_calibrated"] = _sigmoid_stable(z / float(T))
    return out


def apply_dynamic_scores(
    dynamic: DynamicGRMModel,
    new_visits: pd.DataFrame,
) -> pd.DataFrame:
    """Assign new visits to the dynamic state vocabulary and emit per-visit scores.

    Uses the most-recent stored G^(t) (i.e. the largest end_day <= visit day).
    """

    if dynamic.state_preprocessor is None or dynamic.state_kmeans is None:
        raise RuntimeError(
            "Loaded dynamic model has no fitted state preprocessor or KMeans. "
            "Inference under state_source='true_regime' requires true_regime_id in the input visits "
            "and is not handled by this script."
        )

    feature_columns = dynamic.feature_columns
    missing = [c for c in feature_columns if c not in new_visits.columns]
    if missing:
        # Allow dynamic_state features that are derived (delta1, roll3_mean, roll3_std).
        derived = [c for c in missing if c.endswith(("_delta1", "_roll3_mean", "_roll3_std"))]
        if derived and dynamic.state_source == "kmeans_dynamic":
            raise NotImplementedError(
                "kmeans_dynamic state assignment requires recomputing trajectory features "
                "(delta1 / roll3) on incoming visits; not implemented in this minimal predictor."
            )
        raise ValueError(f"Missing required state-feature columns: {missing}")

    X_state = dynamic.state_preprocessor.transform(new_visits[feature_columns].to_numpy(float))
    state_ids = dynamic.state_kmeans.predict(X_state).astype(int)

    # Soft weights with the persisted sigma so they match training.
    diff = X_state[:, None, :] - dynamic.state_centroids[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    sigma = max(float(dynamic.soft_sigma), 1e-9)
    weights = np.exp(-(dist**2) / (2.0 * sigma**2))
    row_sums = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    soft = weights / row_sums

    g_days = sorted(dynamic.global_g_matrices)
    g_days_arr = np.array(g_days, dtype=int)

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(new_visits.itertuples(index=False)):
        day = int(getattr(row, "day"))
        sid = int(state_ids[i])
        eligible = g_days_arr[g_days_arr <= day]
        if eligible.size == 0:
            out_rows.append(
                {
                    "state_id": sid,
                    "matrix_day": None,
                    "self_resonance": float("nan"),
                    "soft_self_resonance": float("nan"),
                    "top1_next_state": -1,
                    "top1_next_state_prob": float("nan"),
                }
            )
            continue
        end_day = int(eligible.max())
        G = dynamic.global_g_matrices[end_day]
        P = dynamic.grm_transition_matrices[end_day]
        diag_g = np.diag(G)
        out_rows.append(
            {
                "state_id": sid,
                "matrix_day": end_day,
                "self_resonance": float(G[sid, sid]),
                "soft_self_resonance": float(soft[i] @ diag_g),
                "top1_next_state": int(np.argmax(P[sid])),
                "top1_next_state_prob": float(np.max(P[sid])),
            }
        )
    return pd.DataFrame(out_rows)


def main() -> None:
    args = parse_args()

    print(f"[load] static model from {args.static_model}")
    static = load_static_model(Path(args.static_model))
    dynamic: Optional[DynamicGRMModel] = None
    if args.dynamic_model:
        print(f"[load] dynamic model from {args.dynamic_model}")
        dynamic = load_dynamic_model(Path(args.dynamic_model), static_model_dir=Path(args.static_model))

    print(f"[load] new visits from {args.visits}")
    visits = pd.read_csv(args.visits)
    missing_obs = [c for c in OBSERVATION_NAMES if c not in visits.columns]
    if missing_obs:
        raise ValueError(f"Input visits missing observation columns: {missing_obs}")
    if "subject_id" not in visits.columns or "day" not in visits.columns:
        raise ValueError("Input visits must contain subject_id and day columns.")

    X_obs = static.obs_preprocessor.transform(visits[OBSERVATION_NAMES].to_numpy(float))

    if args.projection == "surrogate":
        print(f"[step] Projecting {len(visits)} new visits via the Ridge embedding surrogate")
        grm_coords = surrogate_grm_coordinates(static, X_obs)
    else:
        print(f"[step] Nyström-extending GRM basis to {len(visits)} new visits")
        grm_coords = nystrom_grm_coordinates(static, X_obs, n_neighbors=args.n_neighbors)

    out = visits[["subject_id", "day"]].copy()
    for j in range(grm_coords.shape[1]):
        out[f"grm_mode_{j + 1}"] = grm_coords[:, j]

    heads = apply_static_heads(static, grm_coords)
    for k, v in heads.items():
        out[k] = v

    if dynamic is not None:
        print("[step] Computing dynamic state scores")
        dyn_df = apply_dynamic_scores(dynamic, visits)
        for col in dyn_df.columns:
            out[col] = dyn_df[col].to_numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {len(out)} predictions to {out_path}")


if __name__ == "__main__":
    main()
