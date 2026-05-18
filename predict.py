from __future__ import annotations

"""
Out-of-sample inference for persisted GRM-TCM models.

Usage:
    uv run python predict.py --visits NEW.csv \\
        --static-model grm_tcm_results/model \\
        [--dynamic-model grm_tcm_dynamic/model] \\
        --out predictions.csv

Pipeline:
  1. Load static GRM model (preprocessor, eigenbasis, KNN index, regressors).
  2. Apply obs_preprocessor to the new visits' observation columns.
  3. Nyström-extend the spectral basis onto new visits, then weight by 1/(1+rho^2*lambda).
  4. Apply ridge/logistic heads for next_day_score / flare probability.
  5. (Optional) Apply state preprocessor + KMeans from the dynamic model, look up the
     most-recent G^(t) <= visit day, emit self-resonance / soft-self-resonance and
     top-1 next-state transition probability.

Nyström extension uses feature-only edges (KNN + RBF) against training visits. Temporal
and treatment edges from the training graph cannot be reconstructed for unseen visits,
so the extension is an approximation: defensible for visits feature-close to training
data, increasingly noisy for far-out points. No retraining is performed.
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
    parser.add_argument("--n-neighbors", type=int, default=12, help="Neighbors used in Nyström extension.")
    return parser.parse_args()


def nystrom_grm_coordinates(
    static: StaticGRMModel,
    X_new_scaled: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Project new visits into the persisted GRM basis via Nyström extension.

    Returns coordinates of shape (m, n_modes), already weighted by 1/(1 + rho^2 * lambda).
    """

    if static.nn_index is None:
        raise RuntimeError(
            "Loaded static model has no KNN index. Nyström extension requires graph_mode "
            "including feature_only / feature_temporal / feature_temporal_treatment."
        )
    if static.knn_sigma is None:
        raise RuntimeError("Loaded static model is missing knn_sigma; cannot recompute RBF weights.")

    psi = static.eigenvectors  # (n_train, n_modes)
    lambdas = static.eigenvalues  # (n_modes,)
    sigma = float(static.knn_sigma)
    rho = float(static.rho)

    # Request one extra neighbor in case the query coincides with a training row, in which
    # case sklearn returns it at distance 0. We drop any zero-distance match because the
    # training graph excludes self-loops (W.setdiag(0)).
    k = min(int(n_neighbors) + 1, static.nn_index.n_samples_fit_)
    distances, indices = static.nn_index.kneighbors(X_new_scaled, n_neighbors=k)
    self_mask = distances <= 1e-12
    if self_mask.any():
        # Drop the leftmost zero-distance entry per row; keep the remaining n_neighbors.
        keep = np.ones_like(distances, dtype=bool)
        for i in range(distances.shape[0]):
            zero_idx = np.where(distances[i] <= 1e-12)[0]
            if zero_idx.size > 0:
                keep[i, zero_idx[0]] = False
        # Compact rows: take exactly n_neighbors entries from each row.
        new_d = np.empty((distances.shape[0], int(n_neighbors)), dtype=distances.dtype)
        new_i = np.empty((distances.shape[0], int(n_neighbors)), dtype=indices.dtype)
        for i in range(distances.shape[0]):
            mask_row = keep[i]
            picks = np.where(mask_row)[0][: int(n_neighbors)]
            new_d[i] = distances[i, picks]
            new_i[i] = indices[i, picks]
        distances, indices = new_d, new_i
    else:
        distances = distances[:, : int(n_neighbors)]
        indices = indices[:, : int(n_neighbors)]
    w = np.exp(-(distances**2) / (2.0 * sigma**2))  # (m, n_neighbors)

    m = X_new_scaled.shape[0]
    coords = np.zeros((m, psi.shape[1]), dtype=float)

    if static.normalized:
        if static.train_degrees is None:
            raise RuntimeError(
                "Normalized-Laplacian Nyström extension requires persisted train_degrees. "
                "Re-run the trainer to regenerate grm_basis.npz with degrees included."
            )
        d_train = static.train_degrees  # (n_train,)
        inv_sqrt_d_train = 1.0 / np.sqrt(np.maximum(d_train, 1e-12))
        # New-point degree under feature-only edges; matches the form of the training W rows
        # (training W is symmetric so its row sums include in/out contributions; we approximate
        # the new-visit row by the feature-edge contribution only).
        d_new = w.sum(axis=1)  # (m,)
        d_new = np.maximum(d_new, 1e-12)
        inv_sqrt_d_new = 1.0 / np.sqrt(d_new)
        # mu_k = 1 - lambda_k is eigenvalue of normalized adjacency
        mu = 1.0 - lambdas
        # ψ_new[k] = (1 / mu_k) * (1 / sqrt(d_new)) * sum_i (w_i / sqrt(d_i)) * psi[i, k]
        scaled_psi = psi * inv_sqrt_d_train.reshape(-1, 1)  # (n_train, n_modes)
        for j in range(m):
            row_w = w[j]
            idx = indices[j]
            contribution = scaled_psi[idx].T @ row_w  # (n_modes,)
            safe_mu = np.where(np.abs(mu) > 1e-12, mu, np.sign(mu) * 1e-12 + 1e-12)
            coords[j] = contribution * inv_sqrt_d_new[j] / safe_mu
    else:
        # Unnormalized: L = D - W, so L psi = lambda psi
        # New row: d_new * psi_new - sum_i w_i psi[i] = lambda_k * psi_new
        # => psi_new[k] = sum_i w_i psi[i,k] / (d_new - lambda_k)
        d_new = w.sum(axis=1)
        for j in range(m):
            row_w = w[j]
            idx = indices[j]
            contribution = psi[idx].T @ row_w  # (n_modes,)
            denom = d_new[j] - lambdas
            denom = np.where(np.abs(denom) > 1e-9, denom, 1e-9)
            coords[j] = contribution / denom

    spectral_weights = 1.0 / (1.0 + (rho**2) * lambdas)
    return coords * spectral_weights.reshape(1, -1)


def apply_static_heads(static: StaticGRMModel, grm_coords: np.ndarray) -> Dict[str, np.ndarray]:
    """Run the persisted ridge/logistic heads against new GRM coordinates."""

    out: Dict[str, np.ndarray] = {}
    if static.ridge_reg is not None:
        out["pred_next_day_score"] = static.ridge_reg.predict(grm_coords)
    if static.logistic_clf is not None:
        out["pred_flare_prob"] = static.logistic_clf.predict_proba(grm_coords)[:, 1]
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
