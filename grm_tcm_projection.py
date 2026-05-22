from __future__ import annotations

"""
Out-of-sample projection helpers for the static GRM-TCM pipeline.

These are array-based functions that operate on raw fitted state (eigenpairs,
KNN index, sigma, train_degrees, rho, normalized flag). Both predict.py and the
inductive-evaluation path in the trainer call into them so the math lives in
one place.
"""

from typing import Any, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _trim_self_match(
    distances: np.ndarray, indices: np.ndarray, n_neighbors: int
) -> Tuple[np.ndarray, np.ndarray]:
    """If sklearn returned a zero-distance self-match per row, drop it.

    Training graphs use W.setdiag(0), so the new-visit row should not include a
    self-loop weight even when the query matches a training row exactly.
    """

    self_mask = distances <= 1e-12
    if not self_mask.any():
        return distances[:, :n_neighbors], indices[:, :n_neighbors]
    new_d = np.empty((distances.shape[0], n_neighbors), dtype=distances.dtype)
    new_i = np.empty((distances.shape[0], n_neighbors), dtype=indices.dtype)
    for row in range(distances.shape[0]):
        keep = np.ones(distances.shape[1], dtype=bool)
        zero_idx = np.where(distances[row] <= 1e-12)[0]
        if zero_idx.size > 0:
            keep[zero_idx[0]] = False
        picks = np.where(keep)[0][:n_neighbors]
        new_d[row] = distances[row, picks]
        new_i[row] = indices[row, picks]
    return new_d, new_i


def nystrom_extend_arrays(
    X_new_scaled: np.ndarray,
    *,
    nn_index: NearestNeighbors,
    knn_sigma: float,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    rho: float,
    normalized: bool,
    train_degrees: np.ndarray | None,
    n_neighbors: int = 12,
) -> np.ndarray:
    """Project new (already obs_preprocessor-transformed) rows into GRM coordinates.

    Returns (m, n_modes), already weighted by sqrt(1/(1 + rho^2 * lambda_k)).
    With this convention, coordinate inner products approximate the GRM kernel
    G_ij = sum_k psi_k(i) psi_k(j) / (1 + rho^2 lambda_k).

    This is a feature-only Nyström approximation: it reconstructs the new visit's
    KNN+RBF row of W but cannot replay the temporal/treatment edges that the
    training graph may have included. The math is exact only when graph_mode was
    feature_only AND mutual-KNN augmentation didn't change W's row pattern.
    """

    if nn_index is None:
        raise RuntimeError("Nyström extension requires a fitted NN index.")
    sigma = float(knn_sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError(f"Invalid knn_sigma {sigma!r} for Nyström extension.")

    psi = np.asarray(eigenvectors, dtype=float)  # (n_train, n_modes)
    lambdas = np.asarray(eigenvalues, dtype=float)  # (n_modes,)

    k = min(int(n_neighbors) + 1, nn_index.n_samples_fit_)
    distances, indices = nn_index.kneighbors(X_new_scaled, n_neighbors=k)
    distances, indices = _trim_self_match(distances, indices, int(n_neighbors))
    w = np.exp(-(distances**2) / (2.0 * sigma**2))  # (m, n_neighbors)

    m = X_new_scaled.shape[0]
    coords = np.zeros((m, psi.shape[1]), dtype=float)

    if normalized:
        if train_degrees is None:
            raise RuntimeError(
                "Normalized-Laplacian Nyström extension requires train_degrees; "
                "re-train so grm_basis.npz includes degrees."
            )
        d_train = np.asarray(train_degrees, dtype=float)
        inv_sqrt_d_train = 1.0 / np.sqrt(np.maximum(d_train, 1e-12))
        d_new = np.maximum(w.sum(axis=1), 1e-12)
        inv_sqrt_d_new = 1.0 / np.sqrt(d_new)
        mu = 1.0 - lambdas
        scaled_psi = psi * inv_sqrt_d_train.reshape(-1, 1)
        for j in range(m):
            contribution = scaled_psi[indices[j]].T @ w[j]
            safe_mu = np.where(np.abs(mu) > 1e-12, mu, np.sign(mu) * 1e-12 + 1e-12)
            coords[j] = contribution * inv_sqrt_d_new[j] / safe_mu
    else:
        d_new = w.sum(axis=1)
        for j in range(m):
            contribution = psi[indices[j]].T @ w[j]
            denom = d_new[j] - lambdas
            denom = np.where(np.abs(denom) > 1e-9, denom, 1e-9)
            coords[j] = contribution / denom

    spectral_weights = 1.0 / np.sqrt(1.0 + (float(rho) ** 2) * lambdas)
    return coords * spectral_weights.reshape(1, -1)


def surrogate_project(surrogate: Any, X_new_scaled: np.ndarray) -> np.ndarray:
    """Project new rows via a persisted Ridge embedding surrogate.

    Caveat: see CLAUDE.md / predict.py — the surrogate is a deterministic proxy
    for downstream heads, not a faithful spectral reconstruction.
    """

    if surrogate is None:
        raise RuntimeError(
            "No embedding_surrogate available; the model is static-v1 or pre-surrogate."
        )
    return np.asarray(surrogate.predict(X_new_scaled), dtype=float)
