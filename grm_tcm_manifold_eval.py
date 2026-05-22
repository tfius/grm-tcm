from __future__ import annotations

"""Evaluate recovery of the true torus Laplace-Beltrami basis.

This is a geometry benchmark, not a training objective. It compares:
  1. oracle_torus_diffusion: graph built from true torus coordinates
  2. observation_diffusion: graph built from observed clinical channels only
  3. saved_static_grm: embeddings produced by grm_tcm_train.py

The main scores are subspace scores, not one-to-one mode correlations, because
the torus has repeated eigenvalues and eigenvectors can rotate within a
degenerate eigenspace.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import orthogonal_procrustes
from scipy.sparse.linalg import eigsh
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


OBSERVATION_NAMES = [
    "sleep_quality", "hrv", "resting_hr", "body_temp",
    "fatigue", "pain", "appetite", "bowel_quality",
    "mood_calm", "energy", "heaviness", "cold_hot",
]


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    return (x - mu) / np.where(sd > 1e-12, sd, 1.0)


def _orth(x: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(_standardize(x))
    return q


def _subspace_score(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    q = min(a.shape[1], b.shape[1])
    if q < 1:
        return {"mean_cos2": float("nan"), "min_cos": float("nan"), "projection_distance": float("nan")}
    qa = _orth(a[:, :q])
    qb = _orth(b[:, :q])
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    mean_cos2 = float(np.mean(s**2))
    proj_dist = float(np.linalg.norm(qa @ qa.T - qb @ qb.T, ord="fro") / np.sqrt(2.0 * q))
    return {"mean_cos2": mean_cos2, "min_cos": float(np.min(s)), "projection_distance": proj_dist}


def _corr_summary(a: np.ndarray, b: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    a = _standardize(a)
    b = _standardize(b)
    c = (a.T @ b) / max(a.shape[0] - 1, 1)
    rows = []
    for i in range(c.shape[0]):
        j = int(np.nanargmax(np.abs(c[i])))
        rows.append({"mode_index": i + 1, "best_true_mode_index": j + 1, "corr": float(c[i, j]), "abs_corr": float(abs(c[i, j]))})
    df = pd.DataFrame(rows)
    return df, {
        "mean_best_abs_corr": float(df["abs_corr"].mean()),
        "median_best_abs_corr": float(df["abs_corr"].median()),
        "max_best_abs_corr": float(df["abs_corr"].max()),
    }


def _knn_diffusion_modes(X: np.ndarray, *, n_modes: int, n_neighbors: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    k = min(n_neighbors + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
    distances, indices = nn.kneighbors(X)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    positive = distances[distances > 0]
    sigma = float(np.median(positive) if positive.size else 1.0)
    sigma = max(sigma, 1e-9)
    weights = np.exp(-(distances**2) / (2.0 * sigma**2))

    row = np.repeat(np.arange(n), indices.shape[1])
    col = indices.ravel()
    val = weights.ravel()
    K = sparse.coo_matrix((val, (row, col)), shape=(n, n)).tocsr()
    K = K.maximum(K.T)
    K.setdiag(0.0)
    K.eliminate_zeros()

    if alpha != 0.0:
        q = np.maximum(np.asarray(K.sum(axis=1)).ravel(), 1e-12)
        Q = sparse.diags(q ** (-alpha))
        K = Q @ K @ Q
        K.setdiag(0.0)
        K.eliminate_zeros()

    degrees = np.maximum(np.asarray(K.sum(axis=1)).ravel(), 1e-12)
    inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    L = sparse.eye(n, format="csr") - inv_sqrt @ K @ inv_sqrt
    k_eig = min(n_modes + 1, n - 2)
    vals, vecs = eigsh(L, k=k_eig, which="SM")
    order = np.argsort(vals)
    vals = vals[order][1:n_modes + 1]
    vecs = vecs[:, order][:, 1:n_modes + 1]
    return vals, vecs


def _lbo_bands(eigenvalues: np.ndarray, max_modes: int) -> List[Tuple[str, np.ndarray]]:
    vals = np.asarray(eigenvalues, dtype=float)
    # Drop constant mode.
    idx = np.arange(len(vals))[vals > 1e-12]
    idx = idx[:max_modes]
    bands: List[Tuple[str, np.ndarray]] = []
    for lam in sorted(set(np.round(vals[idx], 10))):
        band = idx[np.isclose(vals[idx], lam)]
        if len(band):
            bands.append((f"lambda={float(vals[band[0]]):g}", band))
    return bands


def _score_candidate(name: str, candidate: np.ndarray, true_modes_no_const: np.ndarray, true_eigvals: np.ndarray) -> Dict[str, object]:
    q = min(candidate.shape[1], true_modes_no_const.shape[1])
    corr_df, corr_metrics = _corr_summary(candidate[:, :q], true_modes_no_const[:, :q])
    scores: Dict[str, object] = {
        "candidate": name,
        "n_modes": int(candidate.shape[1]),
        "q_modes_scored": int(q),
        **corr_metrics,
        "cumulative_subspace": {},
        "eigenvalue_bands": {},
    }
    for q_band in [2, 4, 8, q]:
        q_eff = min(q_band, q)
        scores["cumulative_subspace"][f"first_{q_eff}"] = _subspace_score(candidate[:, :q_eff], true_modes_no_const[:, :q_eff])
    for label, band_idx_raw in _lbo_bands(true_eigvals, q + 1):
        # Convert full true-mode index to no-constant column index.
        band_cols = np.asarray([i - 1 for i in band_idx_raw if i > 0 and i - 1 < true_modes_no_const.shape[1]], dtype=int)
        if len(band_cols) == 0:
            continue
        cand_cols = np.arange(min(band_cols[0], candidate.shape[1]), min(band_cols[-1] + 1, candidate.shape[1]))
        if len(cand_cols) == len(band_cols):
            scores["eigenvalue_bands"][label] = _subspace_score(candidate[:, cand_cols], true_modes_no_const[:, band_cols])
    return scores, corr_df


def evaluate_manifold_alignment(
    data_dir: Path,
    results_dir: Path,
    output_dir: Path,
    *,
    n_modes: int = 8,
    n_neighbors: int = 30,
    alpha: float = 1.0,
) -> Dict[str, object]:
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visits = pd.read_csv(data_dir / "visits.csv").sort_values(["subject_id", "day"]).reset_index(drop=True)
    lbo = np.load(data_dir / "true_lbo_eigenmodes.npz", allow_pickle=True)
    true_eigvals = lbo["eigenvalues"].astype(float)
    true_modes = lbo["eigenfunctions"].astype(float)
    true_modes_no_const = true_modes[:, true_eigvals > 1e-12]

    theta_embed = visits[["theta1_sin", "theta1_cos", "theta2_sin", "theta2_cos"]].to_numpy(float)
    _, oracle_modes = _knn_diffusion_modes(theta_embed, n_modes=n_modes, n_neighbors=n_neighbors, alpha=alpha)

    obs = visits[OBSERVATION_NAMES].to_numpy(float)
    obs = SimpleImputer(strategy="median").fit_transform(obs)
    obs = StandardScaler().fit_transform(obs)
    _, obs_modes = _knn_diffusion_modes(obs, n_modes=n_modes, n_neighbors=n_neighbors, alpha=alpha)

    candidates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "oracle_torus_diffusion": (oracle_modes, true_modes_no_const),
        "observation_diffusion": (obs_modes, true_modes_no_const),
    }

    emb_path = results_dir / "grm_visit_embeddings.csv"
    if emb_path.exists():
        emb = pd.read_csv(emb_path)
        mode_cols = [c for c in emb.columns if c.startswith("grm_mode_")]
        lbo_cols = [c for c in visits.columns if c.startswith("true_lbo_mode_")]
        joined = visits[["subject_id", "day"]].merge(
            visits[["subject_id", "day", *lbo_cols]].merge(
                emb[["subject_id", "day", *mode_cols]],
                on=["subject_id", "day"],
                how="inner",
                validate="one_to_one",
            ),
            on=["subject_id", "day"],
            how="inner",
            validate="one_to_one",
        )
        if len(joined) >= 10:
            saved_true_modes = joined[lbo_cols].to_numpy(float)
            # Drop constant true mode.
            saved_true_modes = saved_true_modes[:, 1:] if saved_true_modes.shape[1] > 1 else saved_true_modes
            candidates["saved_static_grm"] = (joined[mode_cols].to_numpy(float)[:, :n_modes], saved_true_modes)

    metrics: Dict[str, object] = {
        "n_visits": int(len(visits)),
        "n_modes": int(n_modes),
        "n_neighbors": int(n_neighbors),
        "alpha_density_normalization": float(alpha),
        "candidates": {},
        "note": (
            "Geometry recovery benchmark. oracle_torus_diffusion should be the upper-bound recovery path. "
            "observation_diffusion tests whether observations preserve manifold geometry. saved_static_grm "
            "tests the current predictive GRM graph; it may predict well while recovering LBO modes less cleanly."
        ),
    }

    for name, (modes, true_for_candidate) in candidates.items():
        candidate_metrics, corr_df = _score_candidate(name, modes, true_for_candidate, true_eigvals)
        metrics["candidates"][name] = candidate_metrics
        corr_df.to_csv(output_dir / f"{name}_mode_correlations.csv", index=False)

    with open(output_dir / "manifold_lbo_alignment_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recovery of true torus LBO eigenspaces.")
    parser.add_argument("--data-dir", default="synthetic_grm_tcm_manifold")
    parser.add_argument("--results-dir", default="grm_tcm_results_manifold")
    parser.add_argument("--output-dir", default="grm_tcm_manifold_eval")
    parser.add_argument("--n-modes", type=int, default=8)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = evaluate_manifold_alignment(
        Path(args.data_dir),
        Path(args.results_dir),
        Path(args.output_dir),
        n_modes=args.n_modes,
        n_neighbors=args.n_neighbors,
        alpha=args.alpha,
    )
    print(json.dumps(out, indent=2))
