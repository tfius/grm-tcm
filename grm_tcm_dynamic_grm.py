from __future__ import annotations

"""
Dynamic GRM diagnostics for the synthetic GRM-TCM benchmark.

This script ports the time-varying propagator pieces from the markets GRM:
- rolling-window resonance matrices G^(t)
- regime-change score ||G^(t) - G^(t-1)||_F
- self-resonance / stuck-state scores G_ii
- GRM-blended transition probabilities

The outputs are synthetic benchmark diagnostics only. They do not prove TCM,
Qi, or a biological mechanism.

Run:
  python grm_tcm_dynamic_grm.py
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/grm_tcm_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/grm_tcm_cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

try:
    import matplotlib
except ModuleNotFoundError as exc:
    if exc.name != "matplotlib":
        raise
    repo_root = Path(__file__).resolve().parent
    venv_dir = repo_root / ".venv"
    venv_python = venv_dir / "bin" / "python3"
    if not venv_python.exists():
        venv_python = venv_dir / "bin" / "python"
    if venv_python.exists() and Path(sys.prefix).resolve() != venv_dir.resolve():
        print(f"[env] matplotlib not found in {sys.executable}; retrying with {venv_python}")
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    raise ModuleNotFoundError(
        "matplotlib is required for dynamic GRM plots. Run `uv sync` or execute `uv run python grm_tcm_dynamic_grm.py`."
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grm_tcm_persistence import (
    manifest_sha,
    save_int_keyed_npz,
    save_joblib,
    write_manifest,
)
from grm_tcm_plot_captions import save_with_caption


DYNAMIC_SCHEMA_VERSION = "dynamic-v1"


OBSERVATION_NAMES = [
    "sleep_quality",
    "hrv",
    "resting_hr",
    "body_temp",
    "fatigue",
    "pain",
    "appetite",
    "bowel_quality",
    "mood_calm",
    "energy",
    "heaviness",
    "cold_hot",
]


@dataclass
class DynamicGRMConfig:
    """Configuration for dynamic GRM analysis."""

    data_dir: str = "synthetic_grm_tcm"
    results_dir: str = "grm_tcm_results"
    output_dir: str = "grm_tcm_dynamic"
    n_states: int = 12
    window_size: int = 14
    step_size: int = 1
    max_modes: int = 8
    energy_threshold: float = 0.95
    alpha: float = 0.65
    feature_weight: float = 0.45
    transition_weight: float = 0.45
    treatment_weight: float = 0.10
    similarity_mode: str = "rbf"
    state_similarity_k: int = 3
    similarity_quantile: float = 0.70
    state_fit_end_day: Optional[int] = None
    state_source: str = "kmeans_observation"
    compare_state_sources: bool = False
    random_seed: int = 42


def parse_args() -> DynamicGRMConfig:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run dynamic GRM diagnostics.")
    parser.add_argument("--data-dir", default="synthetic_grm_tcm")
    parser.add_argument("--results-dir", default="grm_tcm_results")
    parser.add_argument("--output-dir", default="grm_tcm_dynamic")
    parser.add_argument("--n-states", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=14)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--max-modes", type=int, default=8)
    parser.add_argument("--energy-threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--similarity-mode", choices=["rbf", "threshold", "knn"], default="rbf")
    parser.add_argument("--state-similarity-k", type=int, default=3)
    parser.add_argument("--similarity-quantile", type=float, default=0.70)
    parser.add_argument("--state-fit-end-day", type=int, default=None)
    parser.add_argument("--state-source", choices=["kmeans_observation", "kmeans_dynamic", "true_regime"], default="kmeans_observation")
    parser.add_argument("--compare-state-sources", action="store_true")
    args = parser.parse_args()
    return DynamicGRMConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        n_states=args.n_states,
        window_size=args.window_size,
        step_size=args.step_size,
        max_modes=args.max_modes,
        energy_threshold=args.energy_threshold,
        alpha=args.alpha,
        similarity_mode=args.similarity_mode,
        state_similarity_k=args.state_similarity_k,
        similarity_quantile=args.similarity_quantile,
        state_fit_end_day=args.state_fit_end_day,
        state_source=args.state_source,
        compare_state_sources=args.compare_state_sources,
    )


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""

    path.mkdir(parents=True, exist_ok=True)


def read_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    """Read an optional CSV file."""

    if not path.exists():
        print(f"[skip] Missing optional file: {path}")
        return None
    print(f"[load] {path}")
    return pd.read_csv(path)


def mode_columns(df: pd.DataFrame) -> List[str]:
    """Return GRM mode columns in numeric order."""

    cols = [c for c in df.columns if c.startswith("grm_mode_")]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def load_inputs(cfg: DynamicGRMConfig) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load visits, events, and static embeddings."""

    data_dir = Path(cfg.data_dir)
    results_dir = Path(cfg.results_dir)
    visits_path = data_dir / "visits.csv"
    if not visits_path.exists():
        raise FileNotFoundError(f"Missing required file: {visits_path}")
    print(f"[load] {visits_path}")
    visits = pd.read_csv(visits_path).sort_values(["subject_id", "day"]).reset_index(drop=True)
    visits["visit_id"] = np.arange(len(visits))
    events = read_csv_optional(data_dir / "events.csv")
    embeddings = read_csv_optional(results_dir / "grm_visit_embeddings.csv")
    return visits, events, embeddings


def scale_features(frame: pd.DataFrame, columns: Iterable[str]) -> Tuple[np.ndarray, Pipeline, List[str]]:
    """Median-impute and standardize a numeric feature matrix; return matrix, fitted pipe, used columns."""

    cols = [c for c in columns if c in frame.columns]
    if not cols:
        raise ValueError("No requested columns are available for state discretization.")
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    X = pipe.fit_transform(frame[cols].to_numpy(float))
    return X, pipe, cols


def observation_state_features(visits: pd.DataFrame) -> Tuple[np.ndarray, Pipeline, List[str]]:
    """Build observation-only state features. Returns matrix, fitted preprocessor, columns used."""

    cols = [c for c in OBSERVATION_NAMES if c in visits.columns]
    if not cols:
        raise ValueError("No observation features available for state discretization.")
    X, pipe, used = scale_features(visits, cols)
    return X, pipe, used


def dynamic_state_features(visits: pd.DataFrame) -> Tuple[np.ndarray, Pipeline, List[str]]:
    """Build observation trajectory features for dynamic state discretization."""

    cols = [c for c in OBSERVATION_NAMES if c in visits.columns]
    if not cols:
        raise ValueError("No observation features available for dynamic state discretization.")
    work = visits.sort_values(["subject_id", "day"]).copy()
    feature_cols = list(cols)
    for col in cols:
        delta_col = f"{col}_delta1"
        mean_col = f"{col}_roll3_mean"
        std_col = f"{col}_roll3_std"
        grouped = work.groupby("subject_id", sort=False)[col]
        work[delta_col] = grouped.diff().fillna(0.0)
        work[mean_col] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
        work[std_col] = grouped.transform(lambda s: s.rolling(3, min_periods=2).std()).fillna(0.0)
        feature_cols.extend([delta_col, mean_col, std_col])
    work = work.sort_index()
    X, pipe, used = scale_features(work, feature_cols)
    return X, pipe, used


def make_state_features(visits: pd.DataFrame, embeddings: Optional[pd.DataFrame]) -> Tuple[np.ndarray, List[int], List[str]]:
    """Build legacy feature matrix used for fixed state-space discretization."""

    if embeddings is not None:
        modes = mode_columns(embeddings)
        if modes:
            merged = visits[["visit_id", "subject_id", "day"]].merge(
                embeddings[["subject_id", "day"] + modes], on=["subject_id", "day"], how="left"
            )
            complete = merged[modes].notna().all(axis=1)
            if complete.mean() >= 0.5:
                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
                return pipe.fit_transform(merged[modes].to_numpy(float)), merged.index.to_list(), modes

    X, _pipe, cols = observation_state_features(visits)
    return X, visits.index.to_list(), cols


def hard_assignment_centroids(X: np.ndarray, labels: np.ndarray, n_states: int) -> np.ndarray:
    """Compute centroids from assigned rows with global-mean fallback for empty states."""

    global_mean = X.mean(axis=0)
    centroids = []
    for state in range(n_states):
        mask = labels == state
        centroids.append(X[mask].mean(axis=0) if mask.any() else global_mean)
    return np.vstack(centroids)


def assign_states(visits: pd.DataFrame, embeddings: Optional[pd.DataFrame], cfg: DynamicGRMConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    """Assign each visit to a fixed discrete state."""

    X, rows, feature_names = make_state_features(visits, embeddings)
    n_states = min(cfg.n_states, max(2, len(visits) // 5))
    print(f"[step] Discretizing visits into {n_states} states using {len(feature_names)} features")
    fit_mask = np.ones(len(visits), dtype=bool)
    if cfg.state_fit_end_day is not None:
        fit_mask = visits["day"].to_numpy(int) <= cfg.state_fit_end_day
        if fit_mask.sum() < n_states:
            raise ValueError(f"state_fit_end_day leaves fewer rows than n_states: {fit_mask.sum()} < {n_states}")
        print(f"[step] Fitting states on visits through day {cfg.state_fit_end_day}; assigning all visits")
    kmeans = KMeans(n_clusters=n_states, random_state=cfg.random_seed, n_init=20).fit(X[fit_mask])
    labels = kmeans.predict(X)
    out = visits.copy()
    out["state_id"] = labels
    centroids = np.vstack([X[out["state_id"].to_numpy() == state].mean(axis=0) for state in range(n_states)])
    return out, centroids


def soft_state_weights(X: np.ndarray, centroids: np.ndarray, sigma: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Compute soft RBF state assignments for each visit. Returns weights and the sigma used."""

    diff = X[:, None, :] - centroids[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    if sigma is None:
        nonzero = dist[dist > 0]
        sigma = float(np.median(nonzero)) if len(nonzero) else 1.0
    sigma_safe = max(float(sigma), 1e-9)
    weights = np.exp(-(dist**2) / (2.0 * sigma_safe**2))
    row_sums = np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights / row_sums, sigma_safe


@dataclass
class StateModel:
    """Captures everything needed to re-assign new visits to the state vocabulary."""

    source: str
    feature_columns: List[str]
    preprocessor: Optional[Pipeline]
    kmeans: Optional[KMeans]
    centroids: np.ndarray
    soft_sigma: float


def assign_states_and_weights(
    visits: pd.DataFrame, embeddings: Optional[pd.DataFrame], cfg: DynamicGRMConfig
) -> Tuple[pd.DataFrame, StateModel, np.ndarray]:
    """Assign hard states and soft state weights in one shared state vocabulary.

    Returns the visits frame with state_id, a StateModel (capturing preprocessor /
    kmeans / centroids / sigma for out-of-sample reuse), and the soft weights matrix.
    """

    out = visits.copy()
    if cfg.state_source == "kmeans_observation":
        X, preprocessor, feature_names = observation_state_features(visits)
    elif cfg.state_source == "kmeans_dynamic":
        X, preprocessor, feature_names = dynamic_state_features(visits)
    elif cfg.state_source == "true_regime":
        if "true_regime_id" not in visits.columns:
            raise ValueError("--state-source true_regime requires true_regime_id in visits.csv")
        X, preprocessor, feature_names = observation_state_features(visits)
        labels = visits["true_regime_id"].astype(int).to_numpy()
        n_states = int(labels.max()) + 1
        out["state_id"] = labels
        weights = np.zeros((len(out), n_states), dtype=float)
        weights[np.arange(len(out)), labels] = 1.0
        centroids = hard_assignment_centroids(X, labels, n_states)
        # No KMeans was fit; sigma is meaningless for one-hot weights but we still report it.
        _, soft_sigma = soft_state_weights(X, centroids)
        print(f"[step] Using true_regime_id as oracle state vocabulary with {n_states} states")
        out["state_source"] = cfg.state_source
        state_model = StateModel(
            source=cfg.state_source,
            feature_columns=feature_names,
            preprocessor=preprocessor,
            kmeans=None,
            centroids=centroids,
            soft_sigma=soft_sigma,
        )
        return out, state_model, weights
    else:
        raise ValueError(f"Unknown state_source: {cfg.state_source}")

    n_states = min(cfg.n_states, max(2, len(visits) // 5))
    print(
        f"[step] Discretizing visits into {n_states} states using {len(feature_names)} "
        f"{cfg.state_source} features"
    )
    fit_mask = np.ones(len(visits), dtype=bool)
    if cfg.state_fit_end_day is not None:
        fit_mask = visits["day"].to_numpy(int) <= cfg.state_fit_end_day
        if fit_mask.sum() < n_states:
            raise ValueError(f"state_fit_end_day leaves fewer rows than n_states: {fit_mask.sum()} < {n_states}")
        print(f"[step] Fitting states on visits through day {cfg.state_fit_end_day}; assigning all visits")
    kmeans = KMeans(n_clusters=n_states, random_state=cfg.random_seed, n_init=20).fit(X[fit_mask])
    out["state_id"] = kmeans.predict(X)
    out["state_source"] = cfg.state_source
    weights, soft_sigma = soft_state_weights(X, kmeans.cluster_centers_)
    state_model = StateModel(
        source=cfg.state_source,
        feature_columns=feature_names,
        preprocessor=preprocessor,
        kmeans=kmeans,
        centroids=kmeans.cluster_centers_,
        soft_sigma=soft_sigma,
    )
    return out, state_model, weights


def add_event_flags(visits: pd.DataFrame, events: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add per-visit event count columns."""

    out = visits.copy()
    if events is None or events.empty or not {"subject_id", "day", "event_type"}.issubset(events.columns):
        out["treatment_event"] = 0
        return out
    counts = pd.crosstab([events["subject_id"], events["day"]], events["event_type"]).reset_index()
    counts.columns.name = None
    out = out.merge(counts, on=["subject_id", "day"], how="left")
    for col in [c for c in counts.columns if c not in {"subject_id", "day"}]:
        out[col] = out[col].fillna(0).astype(int)
    if "treatment_event" not in out.columns:
        out["treatment_event"] = 0
    return out


def transition_counts(window: pd.DataFrame, n_states: int) -> np.ndarray:
    """Count same-subject one-step transitions between discrete states."""

    counts = np.zeros((n_states, n_states), dtype=float)
    for _, group in window.sort_values(["subject_id", "day"]).groupby("subject_id"):
        states = group["state_id"].to_numpy(int)
        for src, dst in zip(states[:-1], states[1:]):
            counts[src, dst] += 1.0
    return counts


def treatment_counts(window: pd.DataFrame, n_states: int) -> np.ndarray:
    """Count transitions following treatment events."""

    counts = np.zeros((n_states, n_states), dtype=float)
    for _, group in window.sort_values(["subject_id", "day"]).groupby("subject_id"):
        states = group["state_id"].to_numpy(int)
        treatment = group["treatment_event"].to_numpy(int)
        for idx, (src, dst) in enumerate(zip(states[:-1], states[1:])):
            if treatment[idx] > 0:
                counts[src, dst] += 1.0
    return counts


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize a matrix with uniform fallback for empty rows."""

    out = matrix.astype(float).copy()
    row_sums = out.sum(axis=1, keepdims=True)
    empty = row_sums[:, 0] <= 0
    out = np.divide(out, row_sums, out=np.zeros_like(out), where=row_sums > 0)
    if empty.any():
        out[empty, :] = 1.0 / out.shape[1]
    return out


def feature_similarity(centroids: np.ndarray, cfg: DynamicGRMConfig) -> np.ndarray:
    """Compute fixed state-state feature similarity."""

    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    nonzero = dist[dist > 0]
    sigma = float(np.median(nonzero)) if len(nonzero) else 1.0
    sim = np.exp(-(dist**2) / (2.0 * max(sigma, 1e-9) ** 2))
    np.fill_diagonal(sim, 0.0)
    if cfg.similarity_mode == "threshold":
        threshold = float(np.quantile(sim[sim > 0], cfg.similarity_quantile)) if np.any(sim > 0) else 0.0
        sim = np.where(sim >= threshold, sim, 0.0)
    elif cfg.similarity_mode == "knn":
        keep = np.zeros_like(sim, dtype=bool)
        k = min(max(cfg.state_similarity_k, 1), max(sim.shape[0] - 1, 1))
        for i in range(sim.shape[0]):
            idx = np.argsort(sim[i])[-k:]
            keep[i, idx] = True
        keep = keep | keep.T
        sim = np.where(keep, sim, 0.0)
    return sim


def data_scale(window: pd.DataFrame) -> float:
    """Derive the GRM regularization scale from within-window observation variability."""

    cols = [c for c in OBSERVATION_NAMES if c in window.columns]
    if not cols:
        return 1.0
    values = SimpleImputer(strategy="median").fit_transform(window[cols].to_numpy(float))
    scaled = StandardScaler().fit_transform(values)
    scale = float(np.nanmedian(np.nanstd(scaled, axis=0)))
    return max(scale, 0.25)


def build_weight_matrix(window: pd.DataFrame, centroids: np.ndarray, cfg: DynamicGRMConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Build a rolling multi-relational state graph and empirical transition matrix."""

    n_states = len(centroids)
    feature_w = feature_similarity(centroids, cfg)
    trans = transition_counts(window, n_states)
    treat = treatment_counts(window, n_states)
    trans_sym = 0.5 * (row_normalize(trans) + row_normalize(trans).T)
    treat_sym = 0.5 * (row_normalize(treat) + row_normalize(treat).T)
    W = cfg.feature_weight * feature_w + cfg.transition_weight * trans_sym + cfg.treatment_weight * treat_sym
    np.fill_diagonal(W, 0.0)
    return W, trans


def spectral_grm(W: np.ndarray, r_s: float, cfg: DynamicGRMConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """Compute the rolling GRM resonance matrix."""

    degrees = np.maximum(W.sum(axis=1), 1e-12)
    inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L = np.eye(W.shape[0]) - inv_sqrt @ W @ inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = eigenvalues[1:]
    if len(positive) == 0:
        selected = 1
        cumulative_energy = np.array([1.0])
    else:
        energy = positive / max(float(positive.sum()), 1e-12)
        cumulative_energy = np.cumsum(energy)
        selected = int(np.searchsorted(np.cumsum(energy), cfg.energy_threshold) + 1)
        selected = max(1, min(selected, cfg.max_modes, len(positive)))
    idx = slice(1, selected + 1)
    lambdas = eigenvalues[idx]
    psi = eigenvectors[:, idx]
    weights = 1.0 / (1.0 + (r_s**2) * lambdas)
    G = (psi * weights.reshape(1, -1)) @ psi.T
    return G, lambdas, psi, selected, cumulative_energy


def grm_transition_matrix(G: np.ndarray, T_counts: np.ndarray, alpha: float) -> np.ndarray:
    """Blend empirical Markov transitions with row-normalized GRM influence."""

    T = row_normalize(T_counts)
    G_pos = G - np.nanmin(G)
    np.fill_diagonal(G_pos, np.maximum(np.diag(G_pos), 0.0))
    G_norm = row_normalize(G_pos)
    return alpha * T + (1.0 - alpha) * G_norm


def rolling_windows(visits: pd.DataFrame, cfg: DynamicGRMConfig) -> List[Tuple[int, pd.DataFrame]]:
    """Return rolling windows keyed by end day."""

    days = sorted(int(x) for x in visits["day"].dropna().unique())
    windows = []
    for end_day in days:
        start_day = end_day - cfg.window_size + 1
        if start_day < min(days) or (end_day - min(days)) % cfg.step_size != 0:
            continue
        window = visits[(visits["day"] >= start_day) & (visits["day"] <= end_day)].copy()
        if window["day"].nunique() >= max(3, cfg.window_size // 2):
            windows.append((end_day, window))
    return windows


def subject_rolling_windows(subject_visits: pd.DataFrame, cfg: DynamicGRMConfig) -> List[Tuple[int, pd.DataFrame]]:
    """Return rolling windows for one subject."""

    days = sorted(int(x) for x in subject_visits["day"].dropna().unique())
    if not days:
        return []
    windows = []
    min_day = min(days)
    for end_day in days:
        start_day = end_day - cfg.window_size + 1
        if start_day < min_day or (end_day - min_day) % cfg.step_size != 0:
            continue
        window = subject_visits[(subject_visits["day"] >= start_day) & (subject_visits["day"] <= end_day)].copy()
        if window["day"].nunique() >= max(3, cfg.window_size // 2):
            windows.append((end_day, window))
    return windows


def restricted_submatrix(G: np.ndarray, active_states: set[int], union_states: List[int]) -> np.ndarray:
    """Return a zero-filled matrix over union_states restricted to active_states."""

    out = np.zeros((len(union_states), len(union_states)), dtype=float)
    active = [state for state in union_states if state in active_states]
    if not active:
        return out
    positions = [union_states.index(state) for state in active]
    sub = G[np.ix_(active, active)]
    for i, pos_i in enumerate(positions):
        for j, pos_j in enumerate(positions):
            out[pos_i, pos_j] = sub[i, j]
    return out


def compute_subject_dynamic_scores(
    visits: pd.DataFrame,
    state_weights: np.ndarray,
    global_g_matrices: Dict[int, np.ndarray],
    grm_matrices: Dict[int, np.ndarray],
    markov_matrices: Dict[int, np.ndarray],
    cfg: DynamicGRMConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute subject-conditioned scores by restricting population G to visited states."""

    score_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []

    for subject_id, subject in visits.sort_values(["subject_id", "day"]).groupby("subject_id"):
        previous_G: Optional[np.ndarray] = None
        previous_states: set[int] = set()
        subject_soft_values: List[float] = []

        for end_day, window in subject_rolling_windows(subject, cfg):
            available = [d for d in global_g_matrices if d <= end_day]
            if not available:
                continue
            matrix_day = available[-1]
            G = global_g_matrices[matrix_day]

            current = window[window["day"] == end_day]
            if current.empty:
                continue
            row = current.iloc[-1]
            state_id = int(row["state_id"])
            active_states = set(int(x) for x in window["state_id"].dropna().unique())
            if previous_G is None:
                regime_score = float("nan")
            else:
                union_states = sorted(active_states | previous_states)
                current_restricted = restricted_submatrix(G, active_states, union_states)
                previous_restricted = restricted_submatrix(previous_G, previous_states, union_states)
                regime_score = float(np.linalg.norm(current_restricted - previous_restricted, ord="fro"))
            previous_G = G.copy()
            previous_states = active_states
            visit_weights = state_weights[int(row["visit_id"])]
            soft_self = float(visit_weights @ np.diag(G))
            subject_soft_values.append(soft_self)
            score_rows.append(
                {
                    "visit_id": int(row["visit_id"]),
                    "subject_id": int(subject_id),
                    "day": int(end_day),
                    "state_id": state_id,
                    "subject_regime_change_score": regime_score,
                    "subject_self_resonance": float(G[state_id, state_id]),
                    "subject_soft_self_resonance": soft_self,
                    "subject_transition_entropy": float(
                        (-grm_matrices[matrix_day][state_id] * np.log2(np.maximum(grm_matrices[matrix_day][state_id], 1e-12))).sum()
                    ),
                    "n_active_states": int(len(active_states)),
                    "matrix_day": int(matrix_day),
                }
            )

        subject_by_day = {int(r.day): int(r.state_id) for r in subject.itertuples(index=False)}
        matrix_days = sorted(grm_matrices)
        for row in subject.itertuples(index=False):
            day = int(row.day)
            available = [d for d in matrix_days if d <= day]
            if not available or day + 1 not in subject_by_day:
                continue
            end_day = available[-1]
            src = int(row.state_id)
            target = subject_by_day[day + 1]
            P_grm = grm_matrices[end_day]
            P_markov = markov_matrices[end_day]
            transition_rows.append(
                {
                    "visit_id": int(row.visit_id),
                    "subject_id": int(subject_id),
                    "day": day,
                    "source_state": src,
                    "next_state": target,
                    "pred_state_grm": int(np.argmax(P_grm[src])),
                    "pred_state_markov": int(np.argmax(P_markov[src])),
                    "predicted_probability_grm": float(np.max(P_grm[src])),
                    "predicted_probability_markov": float(np.max(P_markov[src])),
                    "target_probability_grm": float(P_grm[src, target]),
                    "target_probability_markov": float(P_markov[src, target]),
                    "window_end_day": end_day,
                }
            )

        if subject_soft_values:
            subject_rows.append(
                {
                    "subject_id": int(subject_id),
                    "mean_subject_soft_self_resonance": float(np.mean(subject_soft_values)),
                    "max_subject_soft_self_resonance": float(np.max(subject_soft_values)),
                    "n_subject_dynamic_windows": int(len(subject_soft_values)),
                }
            )

    return pd.DataFrame(score_rows), pd.DataFrame(transition_rows), pd.DataFrame(subject_rows)


def evaluate_transition_predictions(
    visits: pd.DataFrame,
    grm_matrices: Dict[int, np.ndarray],
    markov_matrices: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """Evaluate next-state prediction using Markov-only and GRM-blended transitions."""

    rows: List[Dict[str, Any]] = []
    by_key = {(int(r.subject_id), int(r.day)): int(r.state_id) for r in visits.itertuples(index=False)}
    day_to_window = sorted(grm_matrices)
    for row in visits.itertuples(index=False):
        day = int(row.day)
        available = [d for d in day_to_window if d <= day]
        if not available:
            continue
        next_key = (int(row.subject_id), day + 1)
        if next_key not in by_key:
            continue
        end_day = available[-1]
        P_grm = grm_matrices[end_day]
        P_markov = markov_matrices[end_day]
        src = int(row.state_id)
        target = by_key[next_key]
        rows.append(
            {
                "visit_id": int(row.visit_id),
                "subject_id": int(row.subject_id),
                "day": day,
                "source_state": src,
                "next_state": target,
                "pred_state_grm": int(np.argmax(P_grm[src])),
                "pred_state_markov": int(np.argmax(P_markov[src])),
                "predicted_probability_grm": float(np.max(P_grm[src])),
                "predicted_probability_markov": float(np.max(P_markov[src])),
                "target_probability_grm": float(P_grm[src, target]),
                "target_probability_markov": float(P_markov[src, target]),
                "window_end_day": end_day,
            }
        )
    return pd.DataFrame(rows)


def safe_auc(y_true: pd.Series, score: pd.Series) -> float:
    """Compute ROC-AUC with one-class guard."""

    pair = pd.concat([y_true, score], axis=1).dropna()
    if pair.empty or pair.iloc[:, 0].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(pair.iloc[:, 0].astype(int), pair.iloc[:, 1].astype(float)))


def transition_metrics(transition_df: pd.DataFrame) -> Dict[str, float]:
    """Compute top-1 and log-loss-style transition metrics."""

    if transition_df.empty:
        return {
            "grm_transition_accuracy": float("nan"),
            "markov_transition_accuracy": float("nan"),
            "transition_accuracy_lift": float("nan"),
            "grm_transition_log_loss": float("nan"),
            "markov_transition_log_loss": float("nan"),
            "transition_log_loss_lift": float("nan"),
            "grm_transition_brier": float("nan"),
            "markov_transition_brier": float("nan"),
            "transition_brier_lift": float("nan"),
            "grm_transition_ece": float("nan"),
            "markov_transition_ece": float("nan"),
            "transition_ece_lift": float("nan"),
        }
    grm_accuracy = float(accuracy_score(transition_df["next_state"], transition_df["pred_state_grm"]))
    markov_accuracy = float(accuracy_score(transition_df["next_state"], transition_df["pred_state_markov"]))
    grm_loss = float(-np.mean(np.log(np.maximum(transition_df["target_probability_grm"].to_numpy(float), 1e-12))))
    markov_loss = float(-np.mean(np.log(np.maximum(transition_df["target_probability_markov"].to_numpy(float), 1e-12))))
    y_grm = (transition_df["pred_state_grm"] == transition_df["next_state"]).astype(int)
    y_markov = (transition_df["pred_state_markov"] == transition_df["next_state"]).astype(int)
    p_grm = transition_df["predicted_probability_grm"].astype(float).clip(0.0, 1.0)
    p_markov = transition_df["predicted_probability_markov"].astype(float).clip(0.0, 1.0)
    grm_brier = float(brier_score_loss(y_grm, p_grm))
    markov_brier = float(brier_score_loss(y_markov, p_markov))
    grm_ece = expected_calibration_error(y_grm.to_numpy(int), p_grm.to_numpy(float))
    markov_ece = expected_calibration_error(y_markov.to_numpy(int), p_markov.to_numpy(float))
    return {
        "grm_transition_accuracy": grm_accuracy,
        "markov_transition_accuracy": markov_accuracy,
        "transition_accuracy_lift": grm_accuracy - markov_accuracy,
        "grm_transition_log_loss": grm_loss,
        "markov_transition_log_loss": markov_loss,
        "transition_log_loss_lift": markov_loss - grm_loss,
        "grm_transition_brier": grm_brier,
        "markov_transition_brier": markov_brier,
        "transition_brier_lift": markov_brier - grm_brier,
        "grm_transition_ece": grm_ece,
        "markov_transition_ece": markov_ece,
        "transition_ece_lift": markov_ece - grm_ece,
    }


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error for binary correctness probabilities."""

    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (probabilities >= lo) & (probabilities <= hi)
        else:
            mask = (probabilities >= lo) & (probabilities < hi)
        if not mask.any():
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - confidence)
    return float(ece)


def reliability_table(transition_df: pd.DataFrame, prefix: str, n_bins: int = 10) -> pd.DataFrame:
    """Build reliability rows for transition target probabilities."""

    if transition_df.empty:
        return pd.DataFrame()
    pred_col = f"pred_state_{prefix}"
    prob_col = f"predicted_probability_{prefix}"
    y_true = (transition_df[pred_col] == transition_df["next_state"]).astype(int).to_numpy()
    probabilities = transition_df[prob_col].astype(float).clip(0.0, 1.0).to_numpy()
    rows = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if hi == 1.0:
            mask = (probabilities >= lo) & (probabilities <= hi)
        else:
            mask = (probabilities >= lo) & (probabilities < hi)
        if not mask.any():
            continue
        rows.append(
            {
                "model": prefix,
                "bin": idx,
                "probability_min": float(lo),
                "probability_max": float(hi),
                "n": int(mask.sum()),
                "mean_predicted_probability": float(probabilities[mask].mean()),
                "empirical_accuracy": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_reliability_outputs(pooled: pd.DataFrame, subject: pd.DataFrame) -> pd.DataFrame:
    """Build pooled and subject-level reliability tables."""

    frames = []
    for scope, df in [("pooled", pooled), ("subject", subject)]:
        for model in ["grm", "markov"]:
            rel = reliability_table(df, model)
            if not rel.empty:
                rel.insert(0, "scope", scope)
                frames.append(rel)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hidden_subtype_eta_squared(subject_summary: pd.DataFrame) -> float:
    """Measure association between subject mean resonance and hidden subtype."""

    required = {"hidden_subtype", "mean_subject_soft_self_resonance"}
    if subject_summary.empty or not required.issubset(subject_summary.columns):
        return float("nan")
    df = subject_summary.dropna(subset=["hidden_subtype", "mean_subject_soft_self_resonance"])
    if df.empty or df["hidden_subtype"].nunique() < 2:
        return float("nan")
    values = df["mean_subject_soft_self_resonance"].to_numpy(float)
    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return float("nan")
    ss_between = 0.0
    for _, group in df.groupby("hidden_subtype"):
        group_values = group["mean_subject_soft_self_resonance"].to_numpy(float)
        ss_between += len(group_values) * (float(group_values.mean()) - grand_mean) ** 2
    return float(ss_between / ss_total)


def eta_squared_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> float:
    """Generic eta-squared for a numeric feature grouped by a categorical variable."""

    if df.empty or not {value_col, group_col}.issubset(df.columns):
        return float("nan")
    work = df[[value_col, group_col]].dropna()
    if work.empty or work[group_col].nunique() < 2:
        return float("nan")
    values = work[value_col].to_numpy(float)
    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return float("nan")
    ss_between = 0.0
    for _, group in work.groupby(group_col):
        group_values = group[value_col].to_numpy(float)
        ss_between += len(group_values) * (float(group_values.mean()) - grand_mean) ** 2
    return float(ss_between / ss_total)


def safe_corr(df: pd.DataFrame, a: str, b: str) -> float:
    """Guarded Pearson correlation for two numeric columns."""

    if df.empty or not {a, b}.issubset(df.columns):
        return float("nan")
    pair = df[[a, b]].dropna()
    if len(pair) < 3 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
        return float("nan")
    return float(pair[a].corr(pair[b]))


def state_regime_mapping(visits: pd.DataFrame) -> Dict[int, int]:
    """Map inferred state ids to their modal true regime id."""

    if not {"state_id", "true_regime_id"}.issubset(visits.columns):
        return {}
    mapping: Dict[int, int] = {}
    for state, group in visits.dropna(subset=["state_id", "true_regime_id"]).groupby("state_id"):
        mapping[int(state)] = int(group["true_regime_id"].mode().iloc[0])
    return mapping


def add_true_regime_transition_eval(transition_df: pd.DataFrame, visits: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate inferred-state transition predictions after mapping states to true regimes."""

    if transition_df.empty or "next_true_regime_id" not in visits.columns:
        return transition_df, {
            "grm_true_regime_transition_accuracy": float("nan"),
            "markov_true_regime_transition_accuracy": float("nan"),
            "true_regime_transition_accuracy_lift": float("nan"),
        }
    mapping = state_regime_mapping(visits)
    if not mapping:
        return transition_df, {
            "grm_true_regime_transition_accuracy": float("nan"),
            "markov_true_regime_transition_accuracy": float("nan"),
            "true_regime_transition_accuracy_lift": float("nan"),
        }
    out = transition_df.merge(visits[["visit_id", "next_true_regime_id"]], on="visit_id", how="left")
    out["pred_true_regime_grm"] = out["pred_state_grm"].map(mapping)
    out["pred_true_regime_markov"] = out["pred_state_markov"].map(mapping)
    pair = out.dropna(subset=["next_true_regime_id", "pred_true_regime_grm", "pred_true_regime_markov"])
    if pair.empty:
        metrics = {
            "grm_true_regime_transition_accuracy": float("nan"),
            "markov_true_regime_transition_accuracy": float("nan"),
            "true_regime_transition_accuracy_lift": float("nan"),
        }
    else:
        grm_acc = float(accuracy_score(pair["next_true_regime_id"].astype(int), pair["pred_true_regime_grm"].astype(int)))
        markov_acc = float(accuracy_score(pair["next_true_regime_id"].astype(int), pair["pred_true_regime_markov"].astype(int)))
        metrics = {
            "grm_true_regime_transition_accuracy": grm_acc,
            "markov_true_regime_transition_accuracy": markov_acc,
            "true_regime_transition_accuracy_lift": grm_acc - markov_acc,
        }
    return out, metrics


def run_dynamic(cfg: DynamicGRMConfig) -> Dict[str, Any]:
    """Run dynamic GRM analysis and write outputs."""

    output_dir = Path(cfg.output_dir)
    plot_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(plot_dir)

    print("[start] Loading dynamic GRM inputs")
    visits, events, embeddings = load_inputs(cfg)
    subjects = read_csv_optional(Path(cfg.data_dir) / "subjects.csv")
    attractors = read_csv_optional(Path(cfg.data_dir) / "true_attractor_states.csv")
    visits, state_model, state_weights = assign_states_and_weights(visits, embeddings, cfg)
    centroids = state_model.centroids
    visits = add_event_flags(visits, events)

    print("[step] Computing rolling resonance matrices")
    windows = rolling_windows(visits, cfg)
    regime_rows: List[Dict[str, Any]] = []
    self_rows: List[Dict[str, Any]] = []
    energy_rows: List[Dict[str, Any]] = []
    transition_matrices: Dict[int, np.ndarray] = {}
    markov_matrices: Dict[int, np.ndarray] = {}
    global_g_matrices: Dict[int, np.ndarray] = {}
    spectral_basis_per_window: Dict[int, Dict[str, np.ndarray]] = {}
    previous_G: Optional[np.ndarray] = None

    for end_day, window in windows:
        W, T_counts = build_weight_matrix(window, centroids, cfg)
        r_s = data_scale(window)
        G, eigenvalues, psi, selected_modes, cumulative_energy = spectral_grm(W, r_s, cfg)
        P = grm_transition_matrix(G, T_counts, cfg.alpha)
        P_markov = row_normalize(T_counts)
        transition_matrices[end_day] = P
        markov_matrices[end_day] = P_markov
        global_g_matrices[end_day] = G
        spectral_basis_per_window[end_day] = {
            "lambdas": np.asarray(eigenvalues, dtype=float),
            "psi": np.asarray(psi, dtype=float),
            "r_s": float(r_s),
            "selected_modes": int(selected_modes),
        }
        regime_score = float(np.linalg.norm(G - previous_G, ord="fro")) if previous_G is not None else float("nan")
        previous_G = G
        regime_rows.append(
            {
                "window_end_day": end_day,
                "window_start_day": end_day - cfg.window_size + 1,
                "regime_change_score": regime_score,
                "r_s": r_s,
                "selected_modes": selected_modes,
                "hit_mode_cap": bool(selected_modes >= min(cfg.max_modes, len(cumulative_energy))),
                "energy_at_selected_modes": float(cumulative_energy[selected_modes - 1]) if len(cumulative_energy) else float("nan"),
                "modes_needed_for_threshold_uncapped": int(np.searchsorted(cumulative_energy, cfg.energy_threshold) + 1)
                if len(cumulative_energy)
                else 1,
                "mean_self_resonance": float(np.mean(np.diag(G))),
                "max_self_resonance": float(np.max(np.diag(G))),
                "mean_transition_entropy": float((-P * np.log2(np.maximum(P, 1e-12))).sum(axis=1).mean()),
                "eigenvalues_json": json.dumps([float(x) for x in eigenvalues]),
            }
        )
        for idx, energy_value in enumerate(cumulative_energy, start=1):
            energy_rows.append(
                {
                    "window_end_day": end_day,
                    "window_start_day": end_day - cfg.window_size + 1,
                    "mode_rank": idx,
                    "cumulative_energy": float(energy_value),
                    "within_max_modes": bool(idx <= cfg.max_modes),
                }
            )
        state_self = np.diag(G)
        for row in window.itertuples(index=False):
            visit_weights = state_weights[int(row.visit_id)]
            self_rows.append(
                {
                    "visit_id": int(row.visit_id),
                    "subject_id": int(row.subject_id),
                    "day": int(row.day),
                    "window_end_day": end_day,
                    "state_id": int(row.state_id),
                    "self_resonance": float(state_self[int(row.state_id)]),
                    "soft_self_resonance": float(visit_weights @ state_self),
                }
            )

    regime_df = pd.DataFrame(regime_rows)
    self_df = pd.DataFrame(self_rows)
    energy_df = pd.DataFrame(energy_rows)
    if not self_df.empty:
        self_df = (
            self_df.sort_values(["visit_id", "window_end_day"])
            .groupby("visit_id", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        visits = visits.merge(self_df[["visit_id", "self_resonance", "soft_self_resonance"]], on="visit_id", how="left")
    else:
        visits["self_resonance"] = np.nan
        visits["soft_self_resonance"] = np.nan

    print("[step] Evaluating dynamic scores")
    visits = visits.merge(
        regime_df[["window_end_day", "regime_change_score"]].rename(columns={"window_end_day": "day"}),
        on="day",
        how="left",
    )
    transition_df = evaluate_transition_predictions(visits, transition_matrices, markov_matrices)
    transition_df, true_regime_transition_metrics = add_true_regime_transition_eval(transition_df, visits)
    pooled_transition_metrics = transition_metrics(transition_df)

    print("[step] Computing subject-conditioned dynamic GRM")
    subject_scores_df, subject_transition_df, subject_summary_df = compute_subject_dynamic_scores(
        visits, state_weights, global_g_matrices, transition_matrices, markov_matrices, cfg
    )
    subject_transition_df, subject_true_regime_transition_metrics = add_true_regime_transition_eval(subject_transition_df, visits)
    subject_transition_metrics = transition_metrics(subject_transition_df)
    if not subject_scores_df.empty:
        visits = visits.merge(
            subject_scores_df[
                [
                    "visit_id",
                    "subject_regime_change_score",
                    "subject_self_resonance",
                    "subject_soft_self_resonance",
                    "subject_transition_entropy",
                ]
            ],
            on="visit_id",
            how="left",
        )
    else:
        visits["subject_regime_change_score"] = np.nan
        visits["subject_self_resonance"] = np.nan
        visits["subject_soft_self_resonance"] = np.nan
        visits["subject_transition_entropy"] = np.nan

    if subjects is not None and not subject_summary_df.empty:
        subject_summary_df = subject_summary_df.merge(subjects[["subject_id", "hidden_subtype"]], on="subject_id", how="left")
    if attractors is not None and not subject_summary_df.empty:
        subject_summary_df = subject_summary_df.merge(attractors, on="subject_id", how="left")
    hidden_eta = hidden_subtype_eta_squared(subject_summary_df)
    reliability_df = make_reliability_outputs(transition_df, subject_transition_df)
    state_regime_confusion = (
        pd.crosstab(visits["state_id"], visits["true_regime"], normalize="index")
        if {"state_id", "true_regime"}.issubset(visits.columns)
        else pd.DataFrame()
    )

    metrics = {
        "config": asdict(cfg),
        "n_windows": int(len(regime_df)),
        "n_states": int(len(centroids)),
        "state_source": cfg.state_source,
        "regime_flare_auc": safe_auc(visits["flare_next_day"], visits["regime_change_score"]) if "flare_next_day" in visits.columns else float("nan"),
        "regime_crash_auc": safe_auc(visits["crash_next_day"], visits["regime_change_score"]) if "crash_next_day" in visits.columns else float("nan"),
        "self_resonance_flare_auc": safe_auc(visits["flare_next_day"], visits["self_resonance"]) if "flare_next_day" in visits.columns else float("nan"),
        "self_resonance_crash_auc": safe_auc(visits["crash_next_day"], visits["self_resonance"]) if "crash_next_day" in visits.columns else float("nan"),
        "soft_self_resonance_flare_auc": safe_auc(visits["flare_next_day"], visits["soft_self_resonance"])
        if "flare_next_day" in visits.columns
        else float("nan"),
        "soft_self_resonance_crash_auc": safe_auc(visits["crash_next_day"], visits["soft_self_resonance"])
        if "crash_next_day" in visits.columns
        else float("nan"),
        "soft_self_resonance_stuck_auc": safe_auc(visits["attractor_state"], visits["soft_self_resonance"])
        if "attractor_state" in visits.columns
        else float("nan"),
        **pooled_transition_metrics,
        **true_regime_transition_metrics,
        "subject_n_windows": int(len(subject_scores_df)),
        "subject_regime_flare_auc": safe_auc(visits["flare_next_day"], visits["subject_regime_change_score"])
        if "flare_next_day" in visits.columns
        else float("nan"),
        "subject_regime_crash_auc": safe_auc(visits["crash_next_day"], visits["subject_regime_change_score"])
        if "crash_next_day" in visits.columns
        else float("nan"),
        "subject_self_resonance_flare_auc": safe_auc(visits["flare_next_day"], visits["subject_self_resonance"])
        if "flare_next_day" in visits.columns
        else float("nan"),
        "subject_self_resonance_crash_auc": safe_auc(visits["crash_next_day"], visits["subject_self_resonance"])
        if "crash_next_day" in visits.columns
        else float("nan"),
        "subject_soft_self_resonance_flare_auc": safe_auc(visits["flare_next_day"], visits["subject_soft_self_resonance"])
        if "flare_next_day" in visits.columns
        else float("nan"),
        "subject_soft_self_resonance_crash_auc": safe_auc(visits["crash_next_day"], visits["subject_soft_self_resonance"])
        if "crash_next_day" in visits.columns
        else float("nan"),
        "subject_soft_self_resonance_hidden_subtype_eta_squared": hidden_eta,
        "subject_mean_soft_resonance_stuck_depleted_correlation": safe_corr(
            subject_summary_df, "mean_subject_soft_self_resonance", "frac_in_stuck_depleted"
        )
        if "frac_in_stuck_depleted" in subject_summary_df.columns
        else float("nan"),
        "subject_mean_soft_resonance_stuck_agitated_correlation": safe_corr(
            subject_summary_df, "mean_subject_soft_self_resonance", "frac_in_stuck_agitated"
        )
        if "frac_in_stuck_agitated" in subject_summary_df.columns
        else float("nan"),
        "frac_in_stuck_depleted_hidden_subtype_eta_squared": eta_squared_by_group(
            subject_summary_df, "frac_in_stuck_depleted", "hidden_subtype"
        )
        if "frac_in_stuck_depleted" in subject_summary_df.columns
        else float("nan"),
        "frac_in_stuck_agitated_hidden_subtype_eta_squared": eta_squared_by_group(
            subject_summary_df, "frac_in_stuck_agitated", "hidden_subtype"
        )
        if "frac_in_stuck_agitated" in subject_summary_df.columns
        else float("nan"),
        "subject_grm_transition_accuracy": subject_transition_metrics["grm_transition_accuracy"],
        "subject_markov_transition_accuracy": subject_transition_metrics["markov_transition_accuracy"],
        "subject_transition_accuracy_lift": subject_transition_metrics["transition_accuracy_lift"],
        "subject_grm_transition_log_loss": subject_transition_metrics["grm_transition_log_loss"],
        "subject_markov_transition_log_loss": subject_transition_metrics["markov_transition_log_loss"],
        "subject_transition_log_loss_lift": subject_transition_metrics["transition_log_loss_lift"],
        "subject_grm_transition_brier": subject_transition_metrics["grm_transition_brier"],
        "subject_markov_transition_brier": subject_transition_metrics["markov_transition_brier"],
        "subject_transition_brier_lift": subject_transition_metrics["transition_brier_lift"],
        "subject_grm_transition_ece": subject_transition_metrics["grm_transition_ece"],
        "subject_markov_transition_ece": subject_transition_metrics["markov_transition_ece"],
        "subject_transition_ece_lift": subject_transition_metrics["transition_ece_lift"],
        "subject_grm_true_regime_transition_accuracy": subject_true_regime_transition_metrics["grm_true_regime_transition_accuracy"],
        "subject_markov_true_regime_transition_accuracy": subject_true_regime_transition_metrics["markov_true_regime_transition_accuracy"],
        "subject_true_regime_transition_accuracy_lift": subject_true_regime_transition_metrics["true_regime_transition_accuracy_lift"],
        "interpretation_guardrail": (
            "Dynamic GRM metrics test rolling resonance, state persistence, and transition propagation in a "
            "synthetic benchmark. They do not prove TCM, Qi, or a biological mechanism."
        ),
    }

    print("[step] Writing dynamic outputs")
    regime_df.to_csv(output_dir / "rolling_regime_scores.csv", index=False)
    energy_df.to_csv(output_dir / "spectral_energy.csv", index=False)
    visits[
        [
            "visit_id",
            "subject_id",
            "day",
            "state_id",
            "self_resonance",
            "soft_self_resonance",
            "regime_change_score",
            "subject_self_resonance",
            "subject_soft_self_resonance",
            "subject_regime_change_score",
            "subject_transition_entropy",
        ]
    ].to_csv(
        output_dir / "self_resonance_scores.csv", index=False
    )
    subject_scores_df.to_csv(output_dir / "subject_dynamic_scores.csv", index=False)
    subject_summary_df.to_csv(output_dir / "subject_resonance_summary.csv", index=False)
    state_regime_confusion.to_csv(output_dir / "inferred_state_true_regime_confusion.csv")
    transition_df.to_csv(output_dir / "grm_transition_predictions.csv", index=False)
    subject_transition_df.to_csv(output_dir / "subject_transition_predictions.csv", index=False)
    reliability_df.to_csv(output_dir / "transition_reliability.csv", index=False)
    state_assignments = visits[["visit_id", "subject_id", "day", "state_id", "state_source"]].copy()
    state_assignments.to_csv(output_dir / "state_assignments.csv", index=False)
    with open(output_dir / "dynamic_grm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_dynamic_model(
        output_dir / "model",
        cfg=cfg,
        state_model=state_model,
        state_weights=state_weights,
        global_g_matrices=global_g_matrices,
        transition_matrices=transition_matrices,
        markov_matrices=markov_matrices,
        spectral_basis_per_window=spectral_basis_per_window,
        regime_df=regime_df,
    )

    save_plots(regime_df, energy_df, visits, reliability_df, state_regime_confusion, subject_summary_df, plot_dir, cfg)
    print_readme(output_dir, metrics)
    return metrics


def save_dynamic_model(
    model_dir: Path,
    *,
    cfg: DynamicGRMConfig,
    state_model: StateModel,
    state_weights: np.ndarray,
    global_g_matrices: Dict[int, np.ndarray],
    transition_matrices: Dict[int, np.ndarray],
    markov_matrices: Dict[int, np.ndarray],
    spectral_basis_per_window: Dict[int, Dict[str, Any]],
    regime_df: pd.DataFrame,
) -> None:
    """Persist the fitted dynamic GRM pipeline alongside its derived CSVs.

    Layout under model_dir:
      - state_preprocessor.joblib (Optional)
      - state_kmeans.joblib (Optional; absent when state_source='true_regime')
      - state_centroids.npy
      - state_metadata.json (source, soft_sigma, feature_columns)
      - state_weights_visit.npy
      - G_matrices.npz / grm_transition_matrices.npz / markov_transition_matrices.npz
      - spectral_basis_per_window.npz  (concatenated lambdas/psi per end_day) + sidecar
      - window_index.parquet
      - manifest.json (with optional static_manifest_sha cross-link)
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if state_model.preprocessor is not None:
        save_joblib(state_model.preprocessor, model_dir / "state_preprocessor.joblib")
    if state_model.kmeans is not None:
        save_joblib(state_model.kmeans, model_dir / "state_kmeans.joblib")
    np.save(model_dir / "state_centroids.npy", state_model.centroids)
    with open(model_dir / "state_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": state_model.source,
                "soft_sigma": float(state_model.soft_sigma),
                "feature_columns": list(state_model.feature_columns),
                "n_states": int(state_model.centroids.shape[0]),
            },
            f,
            indent=2,
        )

    np.save(model_dir / "state_weights_visit.npy", np.asarray(state_weights, dtype=float))

    save_int_keyed_npz(global_g_matrices, model_dir / "G_matrices.npz", prefix="d")
    save_int_keyed_npz(transition_matrices, model_dir / "grm_transition_matrices.npz", prefix="d")
    save_int_keyed_npz(markov_matrices, model_dir / "markov_transition_matrices.npz", prefix="d")

    basis_arrays: Dict[str, np.ndarray] = {}
    basis_sidecar: Dict[str, Dict[str, float]] = {}
    for end_day, entry in spectral_basis_per_window.items():
        basis_arrays[f"lambdas_d{int(end_day)}"] = entry["lambdas"]
        basis_arrays[f"psi_d{int(end_day)}"] = entry["psi"]
        basis_sidecar[str(int(end_day))] = {
            "r_s": float(entry["r_s"]),
            "selected_modes": int(entry["selected_modes"]),
        }
    if basis_arrays:
        np.savez_compressed(model_dir / "spectral_basis_per_window.npz", **basis_arrays)
    with open(model_dir / "spectral_basis_sidecar.json", "w", encoding="utf-8") as f:
        json.dump(basis_sidecar, f, indent=2)

    if not regime_df.empty:
        window_cols = [
            "window_end_day",
            "window_start_day",
            "regime_change_score",
            "r_s",
            "selected_modes",
            "hit_mode_cap",
            "energy_at_selected_modes",
        ]
        regime_df[[c for c in window_cols if c in regime_df.columns]].to_parquet(
            model_dir / "window_index.parquet", index=False
        )

    static_manifest_sha: Optional[str] = None
    static_manifest_path = Path(cfg.results_dir) / "model" / "manifest.json"
    if static_manifest_path.exists():
        static_manifest_sha = manifest_sha(static_manifest_path.parent)

    write_manifest(
        model_dir,
        config=cfg,
        inputs=[
            Path(cfg.data_dir) / "visits.csv",
            Path(cfg.data_dir) / "events.csv",
            Path(cfg.data_dir) / "subjects.csv",
            Path(cfg.data_dir) / "true_attractor_states.csv",
            Path(cfg.results_dir) / "grm_visit_embeddings.csv",
        ],
        schema_version=DYNAMIC_SCHEMA_VERSION,
        random_seed=cfg.random_seed,
        extra={"static_manifest_sha": static_manifest_sha},
    )


def run_state_source_comparison(cfg: DynamicGRMConfig) -> pd.DataFrame:
    """Run dynamic GRM under each state vocabulary and compare key metrics."""

    output_dir = Path(cfg.output_dir)
    plot_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(plot_dir)

    sources = ["kmeans_observation", "kmeans_dynamic", "true_regime"]
    rows: List[Dict[str, Any]] = []
    metric_cols = [
        "regime_flare_auc",
        "soft_self_resonance_flare_auc",
        "subject_regime_flare_auc",
        "subject_soft_self_resonance_flare_auc",
        "transition_accuracy_lift",
        "transition_log_loss_lift",
        "transition_brier_lift",
        "transition_ece_lift",
        "true_regime_transition_accuracy_lift",
        "subject_true_regime_transition_accuracy_lift",
        "soft_self_resonance_stuck_auc",
        "subject_soft_self_resonance_hidden_subtype_eta_squared",
    ]

    for source in sources:
        print(f"\n[compare] Running state_source={source}")
        sub_cfg = replace(
            cfg,
            state_source=source,
            compare_state_sources=False,
            output_dir=str(output_dir / source),
        )
        metrics = run_dynamic(sub_cfg)
        row = {"state_source": source, "n_states": metrics.get("n_states", np.nan)}
        for col in metric_cols:
            row[col] = metrics.get(col, np.nan)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison_path = output_dir / "state_source_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"[write] {comparison_path}")

    plot_cols = [
        "soft_self_resonance_flare_auc",
        "subject_regime_flare_auc",
        "transition_log_loss_lift",
        "true_regime_transition_accuracy_lift",
        "subject_soft_self_resonance_hidden_subtype_eta_squared",
    ]
    available = [c for c in plot_cols if c in comparison.columns and comparison[c].notna().any()]
    if available:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(comparison))
        width = 0.8 / max(len(available), 1)
        for idx, col in enumerate(available):
            ax.bar(x + idx * width - (len(available) - 1) * width / 2.0, comparison[col], width=width, label=col)
        ax.set_xticks(x, labels=comparison["state_source"], rotation=20, ha="right")
        ax.set_title("Dynamic GRM by State Source")
        ax.set_ylabel("Metric value")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        save_with_caption(fig, plot_dir / "state_source_metric_comparison.png", dpi=160)

    print("\nState-source comparison complete.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print("Inspect first:")
    print("  1. state_source_comparison.csv")
    print("  2. plots/state_source_metric_comparison.png")
    print("  3. <state_source>/dynamic_grm_metrics.json")
    print("Interpretation:")
    print("  - kmeans_observation tests whether visible observations define useful states.")
    print("  - kmeans_dynamic tests whether short trajectory features reduce observation aliasing.")
    print("  - true_regime is an oracle ceiling for state vocabulary quality, not a deployable model.")
    return comparison


def save_plots(
    regime_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    visits: pd.DataFrame,
    reliability_df: pd.DataFrame,
    state_regime_confusion: pd.DataFrame,
    subject_summary_df: pd.DataFrame,
    plot_dir: Path,
    cfg: DynamicGRMConfig,
) -> None:
    """Generate dynamic GRM plots."""

    if not regime_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(regime_df["window_end_day"], regime_df["regime_change_score"])
        ax.set_title("Rolling GRM Regime-Change Score")
        ax.set_xlabel("Day")
        ax.set_ylabel("Frobenius change")
        fig.tight_layout()
        save_with_caption(fig, plot_dir / "rolling_regime_change_score.png", dpi=160)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(regime_df["window_end_day"], regime_df["selected_modes"])
        ax.set_title("Energy-Selected GRM Modes")
        ax.set_xlabel("Day")
        ax.set_ylabel("Selected modes")
        fig.tight_layout()
        save_with_caption(fig, plot_dir / "selected_modes_over_time.png", dpi=160)

        if "modes_needed_for_threshold_uncapped" in regime_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(regime_df["window_end_day"], regime_df["modes_needed_for_threshold_uncapped"], label="uncapped")
            ax.plot(regime_df["window_end_day"], regime_df["selected_modes"], label="selected")
            ax.axhline(cfg.max_modes, linestyle="--", linewidth=1, label="max_modes")
            ax.set_title("Selected Modes vs Uncapped Energy Requirement")
            ax.set_xlabel("Day")
            ax.set_ylabel("Modes")
            ax.legend(loc="best")
            fig.tight_layout()
            save_with_caption(fig, plot_dir / "selected_modes_saturation.png", dpi=160)

    if not energy_df.empty:
        pivot = energy_df.pivot(index="window_end_day", columns="mode_rank", values="cumulative_energy")
        limited = pivot[[c for c in pivot.columns if c <= min(cfg.max_modes + 4, int(pivot.columns.max()))]]
        fig, ax = plt.subplots(figsize=(8, 5))
        for day in np.linspace(0, len(limited.index) - 1, num=min(8, len(limited.index)), dtype=int):
            row = limited.iloc[day].dropna()
            ax.plot(row.index, row.values, alpha=0.7)
        ax.axhline(cfg.energy_threshold, linestyle="--", linewidth=1)
        ax.axvline(cfg.max_modes, linestyle="--", linewidth=1)
        ax.set_title("Cumulative Spectral Energy by Window")
        ax.set_xlabel("Mode rank")
        ax.set_ylabel("Cumulative energy")
        fig.tight_layout()
        save_with_caption(fig, plot_dir / "cumulative_spectral_energy.png", dpi=160)

        mean_energy = energy_df.groupby("mode_rank", as_index=False)["cumulative_energy"].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mean_energy["mode_rank"], mean_energy["cumulative_energy"])
        ax.axhline(cfg.energy_threshold, linestyle="--", linewidth=1)
        ax.axvline(cfg.max_modes, linestyle="--", linewidth=1)
        ax.set_title("Mean Cumulative Spectral Energy")
        ax.set_xlabel("Mode rank")
        ax.set_ylabel("Mean cumulative energy")
        fig.tight_layout()
        save_with_caption(fig, plot_dir / "mean_cumulative_spectral_energy.png", dpi=160)

    if {"self_resonance", "global_dysregulation_score"}.issubset(visits.columns):
        pair = visits[["self_resonance", "global_dysregulation_score"]].dropna()
        if not pair.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(pair["self_resonance"], pair["global_dysregulation_score"], s=10, alpha=0.6)
            ax.set_title("Self-Resonance vs Dysregulation")
            ax.set_xlabel("Self-resonance")
            ax.set_ylabel("Global dysregulation score")
            fig.tight_layout()
            save_with_caption(fig, plot_dir / "self_resonance_vs_dysregulation.png", dpi=160)

    if {"subject_regime_change_score", "day"}.issubset(visits.columns):
        grouped = (
            visits.dropna(subset=["subject_regime_change_score"])
            .groupby("day")["subject_regime_change_score"]
            .agg(mean="mean", p90=lambda s: float(np.quantile(s, 0.90)))
            .reset_index()
        )
        if not grouped.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(grouped["day"], grouped["mean"], label="mean")
            ax.plot(grouped["day"], grouped["p90"], label="p90")
            ax.set_title("Subject-Level Regime-Change Score")
            ax.set_xlabel("Day")
            ax.set_ylabel("Frobenius change")
            ax.legend(loc="best")
            fig.tight_layout()
            save_with_caption(fig, plot_dir / "subject_regime_change_score.png", dpi=160)

    if {"subject_self_resonance", "global_dysregulation_score"}.issubset(visits.columns):
        pair = visits[["subject_self_resonance", "global_dysregulation_score"]].dropna()
        if not pair.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(pair["subject_self_resonance"], pair["global_dysregulation_score"], s=10, alpha=0.6)
            ax.set_title("Subject Self-Resonance vs Dysregulation")
            ax.set_xlabel("Subject self-resonance")
            ax.set_ylabel("Global dysregulation score")
            fig.tight_layout()
            save_with_caption(fig, plot_dir / "subject_self_resonance_vs_dysregulation.png", dpi=160)

    if {"soft_self_resonance", "global_dysregulation_score"}.issubset(visits.columns):
        pair = visits[["soft_self_resonance", "global_dysregulation_score"]].dropna()
        if not pair.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(pair["soft_self_resonance"], pair["global_dysregulation_score"], s=10, alpha=0.6)
            ax.set_title("Soft Self-Resonance vs Dysregulation")
            ax.set_xlabel("Soft self-resonance")
            ax.set_ylabel("Global dysregulation score")
            fig.tight_layout()
            save_with_caption(fig, plot_dir / "soft_self_resonance_vs_dysregulation.png", dpi=160)

    if not reliability_df.empty:
        for scope, scoped in reliability_df.groupby("scope"):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
            for model, group in scoped.groupby("model"):
                ax.plot(group["mean_predicted_probability"], group["empirical_accuracy"], marker="o", label=model)
            ax.set_title(f"{scope.title()} Transition Reliability")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Empirical accuracy")
            ax.legend(loc="best")
            fig.tight_layout()
            save_with_caption(fig, plot_dir / f"{scope}_transition_reliability.png", dpi=160)

    if not state_regime_confusion.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(state_regime_confusion.to_numpy(float), aspect="auto")
        ax.set_xticks(np.arange(state_regime_confusion.shape[1]), labels=state_regime_confusion.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(state_regime_confusion.shape[0]), labels=state_regime_confusion.index)
        ax.set_title("Inferred State vs True Regime")
        ax.set_xlabel("True regime")
        ax.set_ylabel("Inferred state")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        save_with_caption(fig, plot_dir / "inferred_state_true_regime_confusion.png", dpi=160)

    if {"hidden_subtype", "frac_in_stuck_depleted", "frac_in_stuck_agitated"}.issubset(subject_summary_df.columns):
        means = subject_summary_df.groupby("hidden_subtype")[["frac_in_stuck_depleted", "frac_in_stuck_agitated"]].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        means.plot(kind="bar", ax=ax)
        ax.set_title("True Stuck Occupancy by Hidden Subtype")
        ax.set_xlabel("hidden_subtype")
        ax.set_ylabel("Mean fraction of days")
        fig.tight_layout()
        save_with_caption(fig, plot_dir / "true_stuck_occupancy_by_hidden_subtype.png", dpi=160)

    if {"mean_subject_soft_self_resonance", "frac_in_any_stuck"}.issubset(subject_summary_df.columns):
        pair = subject_summary_df[["mean_subject_soft_self_resonance", "frac_in_any_stuck"]].dropna()
        if not pair.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(pair["mean_subject_soft_self_resonance"], pair["frac_in_any_stuck"], s=18, alpha=0.75)
            ax.set_title("Subject Resonance vs True Stuck Occupancy")
            ax.set_xlabel("Mean subject soft self-resonance")
            ax.set_ylabel("Fraction in any stuck regime")
            fig.tight_layout()
            save_with_caption(fig, plot_dir / "subject_resonance_vs_true_stuck_occupancy.png", dpi=160)


def print_readme(output_dir: Path, metrics: Dict[str, Any]) -> None:
    """Print concise output guidance."""

    print("\nDynamic GRM complete.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print("Inspect first:")
    print("  1. dynamic_grm_metrics.json")
    print("  2. rolling_regime_scores.csv")
    print("  3. spectral_energy.csv")
    print("  4. self_resonance_scores.csv")
    print("  5. subject_dynamic_scores.csv")
    print("  6. subject_resonance_summary.csv")
    print("  7. inferred_state_true_regime_confusion.csv")
    print("  8. grm_transition_predictions.csv")
    print("  9. subject_transition_predictions.csv")
    print("  10. transition_reliability.csv")
    print("Interpretation:")
    print("  - High regime-change AUC means rolling G changes before flares/crashes.")
    print("  - High self-resonance AUC means attractor/stuck-state strength is predictive.")
    print("  - Transition accuracy tests literal G-as-propagator semantics.")
    print("  - Positive transition lift means GRM blending improves on Markov-only transitions.")
    print("  - Subject-level scores test individual pre-flare instability instead of pooled population shifts.")
    print("  - Synthetic benchmark only; no TCM, Qi, or biological mechanism is proven.")
    print(
        "Key metrics: "
        f"regime_flare_auc={metrics['regime_flare_auc']:.3f}, "
        f"soft_self_resonance_flare_auc={metrics['soft_self_resonance_flare_auc']:.3f}, "
        f"subject_regime_flare_auc={metrics['subject_regime_flare_auc']:.3f}, "
        f"subject_soft_self_resonance_flare_auc={metrics['subject_soft_self_resonance_flare_auc']:.3f}, "
        f"transition_accuracy={metrics['grm_transition_accuracy']:.3f}, "
        f"transition_lift={metrics['transition_accuracy_lift']:.3f}"
    )


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.compare_state_sources:
        run_state_source_comparison(parsed)
    else:
        run_dynamic(parsed)
