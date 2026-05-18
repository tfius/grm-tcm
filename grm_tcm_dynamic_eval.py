from __future__ import annotations

"""
GRM-TCM dynamic eval — focused, falsifiable evaluation of GRM's distinctive claims.

Consumes the persisted static + dynamic models (no recomputation) and produces a
structured certificate file that says, with bootstrap CIs, which GRM-related
claims survive scrutiny on the regime-switching synthetic benchmark.

Run:
  python grm_tcm_dynamic_eval.py
  python grm_tcm_dynamic_eval.py --scope aliased --bootstrap-n 500
  python grm_tcm_dynamic_eval.py --scope transitions,fingerprints,aliased

Outputs (under grm_tcm_dynamic_eval/):
  dynamic_eval_certificates.json     -- structured falsifiable verdicts with CIs
  transition_metrics.csv              -- per-model transition metrics with bootstrap
  transition_reliability.csv          -- reliability curve data per model
  dwell_metrics.csv                   -- total + remaining dwell prediction
  subject_fingerprint_metrics.csv     -- hidden_subtype recovery per feature set
  subject_fingerprints.csv            -- raw subject feature vectors
  aliased_state_analysis.csv          -- T1-T4 results on aliased visits
  ablation_metrics.csv                -- shuffled-time/subjects/random-embed control
  plots/                              -- 7+ matplotlib PNGs

Scientific framing:
  Diagnostic on a known synthetic generator. Not a biological simulator. Not
  evidence for TCM or Qi. Verdicts here describe latent-state recovery,
  attractor fingerprinting, and ontology-mismatch detection on synthetic data.

Methodological caveats (recorded in dynamic_eval_certificates.json["caveats"]):
  - Persisted GRM transition matrices G^(t) were fit on all training subjects,
    not refit per CV fold. Evaluation is subject-CV honest at the prediction
    layer but the GRM model itself has seen all subjects. This is leakage in
    GRM's favor: any FAIL verdict against GRM is therefore conservative.
  - The strong-baseline transition model uses state_id (cluster vocabulary),
    matching what GRM has access to. It does NOT use true_regime_id.
"""

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/grm_tcm_matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from grm_tcm_load import DynamicGRMModel, StaticGRMModel, load_dynamic_model, load_static_model
from grm_tcm_plot_captions import save_with_caption


OBSERVATION_NAMES: List[str] = [
    "sleep_quality", "hrv", "resting_hr", "body_temp", "fatigue", "pain",
    "appetite", "bowel_quality", "mood_calm", "energy", "heaviness", "cold_hot",
]
LATENT_NAMES: List[str] = [
    "vitality_depletion", "stress_activation", "inflammatory_load", "digestive_instability",
]
STUCK_REGIME_IDS: Tuple[int, ...] = (4, 5)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


SCOPES = ("transitions", "dwell", "fingerprints", "aliased", "ablations", "plots", "all")


@dataclass
class DynamicEvalConfig:
    """All knobs for the eval run."""

    data_dir: Path = Path("synthetic_grm_tcm")
    static_model_dir: Path = Path("grm_tcm_results/model")
    dynamic_model_dir: Path = Path("grm_tcm_dynamic/model")
    output_dir: Path = Path("grm_tcm_dynamic_eval")
    scope: Tuple[str, ...] = ("all",)
    seed: int = 42
    bootstrap_n: int = 200
    cv_splits: int = 5
    aliased_k_nn: int = 10
    aliased_min_each: int = 3
    dirichlet_kappa: float = 10.0
    grm_blend_alpha: float = 0.65


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> DynamicEvalConfig:
    """Parse CLI flags into a config."""
    p = argparse.ArgumentParser(description="Dynamic eval of persisted GRM-TCM models.")
    p.add_argument("--data-dir", type=Path, default=Path("synthetic_grm_tcm"))
    p.add_argument("--static-model", type=Path, default=Path("grm_tcm_results/model"))
    p.add_argument("--dynamic-model", type=Path, default=Path("grm_tcm_dynamic/model"))
    p.add_argument("--output-dir", type=Path, default=Path("grm_tcm_dynamic_eval"))
    p.add_argument("--scope", default="all", help=f"Comma-separated subset of {SCOPES}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-n", type=int, default=200)
    p.add_argument("--cv-splits", type=int, default=5)
    args = p.parse_args()
    scope = tuple(s.strip() for s in args.scope.split(",") if s.strip())
    for s in scope:
        if s not in SCOPES:
            raise ValueError(f"Unknown scope: {s}. Choose from {SCOPES}.")
    return DynamicEvalConfig(
        data_dir=args.data_dir,
        static_model_dir=args.static_model,
        dynamic_model_dir=args.dynamic_model,
        output_dir=args.output_dir,
        scope=scope,
        seed=args.seed,
        bootstrap_n=args.bootstrap_n,
        cv_splits=args.cv_splits,
    )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


@dataclass
class EvalSetup:
    """All shared state hydrated once and reused across eval modules."""

    cfg: DynamicEvalConfig
    static: StaticGRMModel
    dynamic: DynamicGRMModel
    visits: pd.DataFrame                # full visits frame with state_id, next_state, etc.
    subjects: pd.DataFrame
    embeddings: np.ndarray              # (n_train, n_modes) — aligned with visit_index
    embedding_visit_index: pd.DataFrame # (visit_id, subject_id, day) for embeddings rows
    state_weights: np.ndarray           # (n_visits, n_states)
    state_assignments: np.ndarray       # (n_visits,) hard argmax
    self_resonance: np.ndarray          # (n_visits,) soft self-resonance
    g_end_day_lookup: np.ndarray        # (n_visits,) end_day of G^(t) used (-1 if none)
    g_days_sorted: np.ndarray           # sorted G end-days available
    rng: np.random.Generator


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def _build_state_columns(visits: pd.DataFrame, state_weights: np.ndarray, n_states: int) -> pd.DataFrame:
    """Attach hard state assignment and per-state weight columns to visits frame."""
    out = visits.copy()
    out["state_id"] = state_weights.argmax(axis=1)
    return out


def _soft_self_resonance(
    state_weights: np.ndarray,
    g_days_sorted: np.ndarray,
    g_matrices: Dict[int, np.ndarray],
    visit_days: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute soft self-resonance per visit and record which window's G was used."""
    n = state_weights.shape[0]
    out = np.full(n, np.nan, dtype=float)
    used = np.full(n, -1, dtype=int)
    if g_days_sorted.size == 0:
        return out, used
    for i in range(n):
        day = int(visit_days[i])
        eligible = g_days_sorted[g_days_sorted <= day]
        if eligible.size == 0:
            continue
        end_day = int(eligible.max())
        G = g_matrices[end_day]
        out[i] = float(state_weights[i] @ np.diag(G))
        used[i] = end_day
    return out, used


def build_setup(cfg: DynamicEvalConfig) -> EvalSetup:
    """Hydrate persisted models and assemble the per-visit analysis frame."""
    print(f"[setup] loading static model from {cfg.static_model_dir}")
    static = load_static_model(cfg.static_model_dir)
    print(f"[setup] loading dynamic model from {cfg.dynamic_model_dir}")
    dynamic = load_dynamic_model(cfg.dynamic_model_dir, static_model_dir=cfg.static_model_dir)

    visits = _read_csv(cfg.data_dir / "visits.csv")
    subjects = _read_csv(cfg.data_dir / "subjects.csv")
    visits = visits.sort_values(["subject_id", "day"]).reset_index(drop=True)

    state_weights = dynamic.state_weights
    if state_weights.shape[0] != len(visits):
        raise ValueError(
            f"state_weights rows ({state_weights.shape[0]}) != visits rows ({len(visits)}). "
            "Re-run the dynamic pipeline against the current visits.csv."
        )
    n_states = state_weights.shape[1]
    visits = _build_state_columns(visits, state_weights, n_states)

    # Within-subject lead of state_id gives next_state; last day per subject = NaN.
    visits["next_state_id"] = visits.groupby("subject_id")["state_id"].shift(-1)
    visits["next_true_regime_id"] = visits.groupby("subject_id")["true_regime_id"].shift(-1)

    embeddings = static.eigenvectors * (1.0 / (1.0 + (static.rho ** 2) * static.eigenvalues)).reshape(1, -1)
    embedding_visit_index = static.visit_index.copy() if static.visit_index is not None else pd.DataFrame(
        {"visit_id": np.arange(embeddings.shape[0])}
    )

    g_days_sorted = np.array(sorted(dynamic.global_g_matrices), dtype=int)
    self_res, g_end_day_lookup = _soft_self_resonance(
        state_weights, g_days_sorted, dynamic.global_g_matrices, visits["day"].to_numpy(int),
    )
    visits["soft_self_resonance"] = self_res
    visits["g_end_day"] = g_end_day_lookup

    return EvalSetup(
        cfg=cfg,
        static=static,
        dynamic=dynamic,
        visits=visits,
        subjects=subjects,
        embeddings=embeddings,
        embedding_visit_index=embedding_visit_index,
        state_weights=state_weights,
        state_assignments=visits["state_id"].to_numpy(int),
        self_resonance=self_res,
        g_end_day_lookup=g_end_day_lookup,
        g_days_sorted=g_days_sorted,
        rng=np.random.default_rng(cfg.seed),
    )


# ---------------------------------------------------------------------------
# Bootstrap + utilities
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    n_boot: int = 200,
    seed: int = 42,
    statistic: str = "mean",
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) bootstrapped over `groups` (or rows)."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    values = values[mask]
    if groups is not None:
        groups = np.asarray(groups)[mask]
    rng = np.random.default_rng(seed)
    point = float(np.mean(values)) if statistic == "mean" else float(np.median(values))
    samples = np.empty(n_boot)
    if groups is None:
        for b in range(n_boot):
            idx = rng.integers(0, len(values), size=len(values))
            samples[b] = float(np.mean(values[idx])) if statistic == "mean" else float(np.median(values[idx]))
    else:
        unique_groups = np.unique(groups)
        index_by_group = {int(g): np.where(groups == g)[0] for g in unique_groups}
        for b in range(n_boot):
            picks = rng.integers(0, len(unique_groups), size=len(unique_groups))
            chosen = np.concatenate([index_by_group[int(unique_groups[p])] for p in picks])
            samples[b] = float(np.mean(values[chosen])) if statistic == "mean" else float(np.median(values[chosen]))
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def cluster_bootstrap_paired(
    subject_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray], Tuple[float, ...]],
    *,
    n_boot: int,
    seed: int,
    label: str = "",
) -> List[Tuple[float, float, float]]:
    """Cluster bootstrap by subject. `metric_fn(idx)` returns a tuple of scalars.

    Returns one (point, ci_low, ci_high) per scalar in the tuple. The point estimate
    uses all rows; CI bounds are 2.5% / 97.5% quantiles across B subject-resamples.
    Skipped iterations (non-finite or exception) are counted and printed when
    `label` is non-empty.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(subject_ids)
    by_subject = {int(s): np.where(subject_ids == s)[0] for s in unique}
    point = metric_fn(np.arange(len(subject_ids)))
    samples: List[Tuple[float, ...]] = []
    n_skipped = 0
    for _ in range(int(n_boot)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_subject[int(s)] for s in chosen])
        try:
            vals = metric_fn(idx)
            if all(np.isfinite(v) for v in vals):
                samples.append(vals)
            else:
                n_skipped += 1
        except Exception:
            n_skipped += 1
            continue
    if label and n_skipped:
        print(f"[bootstrap:{label}] skipped {n_skipped}/{int(n_boot)} iterations (non-finite or raised)")
    if not samples:
        return [(float(p), float("nan"), float("nan")) for p in point]
    arr = np.array(samples)
    return [
        (float(point[i]), float(np.quantile(arr[:, i], 0.025)), float(np.quantile(arr[:, i], 0.975)))
        for i in range(arr.shape[1])
    ]


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    """Multiclass ECE using top-1 predicted probability vs top-1 correctness."""
    if probabilities.ndim != 2 or len(y_true) == 0:
        return float("nan")
    top_prob = probabilities.max(axis=1)
    top_pred = probabilities.argmax(axis=1)
    correct = (top_pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (top_prob > lo) & (top_prob <= hi) if hi < 1.0 else (top_prob > lo) & (top_prob <= hi + 1e-9)
        if mask.sum() == 0:
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(top_prob[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_table(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin accuracy vs confidence for a reliability diagram."""
    top_prob = probabilities.max(axis=1)
    top_pred = probabilities.argmax(axis=1)
    correct = (top_pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (top_prob > lo) & (top_prob <= hi) if hi < 1.0 else (top_prob > lo) & (top_prob <= hi + 1e-9)
        n = int(mask.sum())
        rows.append({
            "bin_low": float(lo), "bin_high": float(hi),
            "n": n,
            "mean_confidence": float(top_prob[mask].mean()) if n else float("nan"),
            "empirical_accuracy": float(correct[mask].mean()) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def _multi_brier(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int) -> float:
    """One-hot Brier score across classes."""
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _topk_accuracy(y_true: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    """Top-k accuracy: is y_true among the k highest-prob classes?"""
    top_k = np.argsort(-probabilities, axis=1)[:, :k]
    return float(np.mean(np.any(top_k == y_true.reshape(-1, 1), axis=1)))


# ---------------------------------------------------------------------------
# Transition models
# ---------------------------------------------------------------------------


def _empirical_markov(
    current: np.ndarray, nxt: np.ndarray, n_states: int, alpha: float = 0.0
) -> np.ndarray:
    """Pooled Laplace-smoothed transition matrix from a (current, next) pair set."""
    M = np.full((n_states, n_states), alpha, dtype=float)
    for c, n in zip(current, nxt):
        M[int(c), int(n)] += 1.0
    M += alpha
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return M / row_sums


def _grm_blended_from_persisted(
    dynamic: DynamicGRMModel, current_states: np.ndarray, end_days: np.ndarray
) -> np.ndarray:
    """Per-visit probability rows using persisted GRM-blended transition matrices.

    For each visit (current_state s, end_day d): return GRM_T[d][s, :].
    Falls back to a uniform distribution where end_day == -1.

    The persisted matrices already encode the Markov+G blend applied during the
    dynamic pipeline run; there is no `alpha` to apply here.
    """
    n_visits = len(current_states)
    n_states = next(iter(dynamic.grm_transition_matrices.values())).shape[0]
    out = np.full((n_visits, n_states), 1.0 / n_states, dtype=float)
    for i in range(n_visits):
        d = int(end_days[i])
        if d < 0 or d not in dynamic.grm_transition_matrices:
            continue
        s = int(current_states[i])
        row = dynamic.grm_transition_matrices[d][s]
        out[i] = row
    return out


def _markov_from_persisted(
    dynamic: DynamicGRMModel, current_states: np.ndarray, end_days: np.ndarray
) -> np.ndarray:
    """Per-visit probability rows using persisted Markov (pre-GRM-blend) matrices."""
    n_visits = len(current_states)
    n_states = next(iter(dynamic.markov_transition_matrices.values())).shape[0]
    out = np.full((n_visits, n_states), 1.0 / n_states, dtype=float)
    for i in range(n_visits):
        d = int(end_days[i])
        if d < 0 or d not in dynamic.markov_transition_matrices:
            continue
        s = int(current_states[i])
        row = dynamic.markov_transition_matrices[d][s]
        out[i] = row
    return out


def _dirichlet_subject_personalized(
    subject_ids: np.ndarray,
    current_states: np.ndarray,
    next_states: np.ndarray,
    eval_idx: np.ndarray,
    train_idx: np.ndarray,
    n_states: int,
    kappa: float,
) -> np.ndarray:
    """Dirichlet posterior on per-subject transitions with a global prior.

    posterior_s[c, j] ∝ kappa * global[c, j] + counts_subject[c, j]
    Fit global on train_idx only to avoid leakage.
    """
    n_states_t = int(n_states)
    global_matrix = _empirical_markov(current_states[train_idx], next_states[train_idx], n_states_t, alpha=1.0)

    subject_counts: Dict[int, np.ndarray] = {}
    for i in train_idx:
        s = int(subject_ids[i])
        if s not in subject_counts:
            subject_counts[s] = np.zeros((n_states_t, n_states_t), dtype=float)
        subject_counts[s][int(current_states[i]), int(next_states[i])] += 1.0

    out = np.zeros((len(eval_idx), n_states_t), dtype=float)
    for k, i in enumerate(eval_idx):
        s = int(subject_ids[i])
        c = int(current_states[i])
        counts = subject_counts.get(s, np.zeros((n_states_t, n_states_t)))
        row = kappa * global_matrix[c] + counts[c]
        row_sum = row.sum()
        out[k] = row / row_sum if row_sum > 0 else np.full(n_states_t, 1.0 / n_states_t)
    return out


def _mnl_baseline_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, n_states: int, seed: int = 42
) -> np.ndarray:
    """Multinomial logistic regression baseline; returns full-class probabilities."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_eval)
    full = np.full((X_eval.shape[0], n_states), 1e-6, dtype=float)
    for i, cls in enumerate(clf.classes_):
        full[:, int(cls)] = prob[:, i]
    full = full / full.sum(axis=1, keepdims=True)
    return full


# ---------------------------------------------------------------------------
# eval_transitions
# ---------------------------------------------------------------------------


def _transition_eval_rows(name: str, y_true: np.ndarray, probs: np.ndarray, n_states: int) -> Dict[str, float]:
    """Return a single-row metric dict for one model on one fold."""
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {
        "model": name,
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, probs, labels=list(range(n_states)))),
        "brier": _multi_brier(y_true, probs, n_states),
        "top1_acc": float(np.mean(probs.argmax(axis=1) == y_true)),
        "top2_acc": _topk_accuracy(y_true, probs, k=2),
        "macro_f1": float(f1_score(y_true, probs.argmax(axis=1), average="macro", zero_division=0)),
        "ece": expected_calibration_error(y_true, probs, n_bins=10),
    }


def _baseline_feature_matrix(visits: pd.DataFrame, n_states: int) -> np.ndarray:
    """Strong baseline F: current state (cluster) one-hot + dwell + delayed loads.

    Uses `state_id` (not `true_regime_id`) so the baseline operates in the same
    vocabulary as GRM. Including `true_regime_id` would give the baseline oracle
    access GRM does not have, making the comparison unfair.
    """
    n = len(visits)
    one_hot = np.zeros((n, n_states), dtype=float)
    one_hot[np.arange(n), visits["state_id"].to_numpy(int)] = 1.0
    extras = []
    for col, default in [
        ("dwell_time", 0.0),
        ("delayed_stress_load", 0.0),
        ("delayed_treatment_load", 0.0),
        ("delayed_recovery_load", 0.0),
        ("latent_instability", 0.0),
    ]:
        if col in visits.columns:
            extras.append(visits[col].fillna(default).to_numpy(float).reshape(-1, 1))
    return np.hstack([one_hot] + extras) if extras else one_hot


def eval_transitions(setup: EvalSetup) -> Dict[str, Any]:
    """Six transition models with subject-grouped CV and bootstrap CIs."""
    cfg = setup.cfg
    visits = setup.visits
    n_states = setup.state_weights.shape[1]

    mask = visits["next_state_id"].notna() & visits["g_end_day"].ge(0)
    eval_df = visits[mask].reset_index(drop=True)
    current = eval_df["state_id"].to_numpy(int)
    nxt = eval_df["next_state_id"].to_numpy(int)
    subj = eval_df["subject_id"].to_numpy(int)
    end_days = eval_df["g_end_day"].to_numpy(int)
    print(f"[transitions] eligible visits: {len(eval_df)} / {len(visits)}")

    baseline_X = _baseline_feature_matrix(eval_df, n_states)

    gkf = GroupKFold(n_splits=cfg.cv_splits)
    fold_rows: List[Dict[str, Any]] = []
    reliability_rows: List[Dict[str, Any]] = []
    per_visit_logloss: Dict[str, List[Tuple[int, float]]] = {}

    grm_persisted_full = _grm_blended_from_persisted(setup.dynamic, current, end_days)
    markov_persisted_full = _markov_from_persisted(setup.dynamic, current, end_days)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(current, nxt, groups=subj)):
        y_test = nxt[test_idx]
        # Empirical Markov fit on train (CV-honest)
        markov_train = _empirical_markov(current[train_idx], nxt[train_idx], n_states, alpha=0.0)
        markov_laplace = _empirical_markov(current[train_idx], nxt[train_idx], n_states, alpha=1.0)
        probs_a = markov_train[current[test_idx]]
        probs_b = markov_laplace[current[test_idx]]
        # GRM-blended from persisted (uses train+test in matrices, but matrices are pre-computed; subject-CV is honest only at evaluation level)
        probs_c = grm_persisted_full[test_idx]
        # Dirichlet subject-personalized
        probs_d = _dirichlet_subject_personalized(subj, current, nxt, test_idx, train_idx, n_states, kappa=cfg.dirichlet_kappa)
        # MNL baseline F
        probs_f = _mnl_baseline_predict(
            baseline_X[train_idx], nxt[train_idx], baseline_X[test_idx], n_states, seed=cfg.seed + fold_idx,
        )
        # Markov from persisted matrices (sanity check vs A)
        probs_persisted_markov = markov_persisted_full[test_idx]

        for name, probs in [
            ("A_markov", probs_a),
            ("B_markov_laplace", probs_b),
            ("C_grm_blended", probs_c),
            ("D_dirichlet_subject", probs_d),
            ("F_mnl_baseline", probs_f),
            ("persisted_markov_sanity", probs_persisted_markov),
        ]:
            row = _transition_eval_rows(name, y_test, probs, n_states)
            row["fold"] = fold_idx
            fold_rows.append(row)
            for j, ll in enumerate(_per_row_log_loss(y_test, probs)):
                per_visit_logloss.setdefault(name, []).append((int(test_idx[j]), float(ll)))
            rel = reliability_table(y_test, probs)
            rel["model"] = name
            rel["fold"] = fold_idx
            reliability_rows.append(rel)

    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    for model, sub in fold_df.groupby("model"):
        for metric in ["log_loss", "brier", "top1_acc", "top2_acc", "macro_f1", "ece"]:
            vals = sub[metric].to_numpy(float)
            point, lo, hi = bootstrap_ci(vals, n_boot=cfg.bootstrap_n, seed=cfg.seed)
            summary_rows.append({"model": model, "metric": metric, "mean": point, "ci_low": lo, "ci_high": hi})
    summary_df = pd.DataFrame(summary_rows)

    reliability_full = pd.concat(reliability_rows, ignore_index=True) if reliability_rows else pd.DataFrame()
    # Aggregate per (model, bin): mean confidence + frequency-weighted mean accuracy
    # across folds. Folds with zero rows in a bin contribute nothing.
    if not reliability_full.empty:
        agg_rows = []
        for (model, bin_lo, bin_hi), grp in reliability_full.groupby(["model", "bin_low", "bin_high"]):
            grp_valid = grp[grp["n"] > 0]
            total_n = int(grp_valid["n"].sum())
            if total_n == 0:
                agg_rows.append({
                    "model": model, "bin_low": float(bin_lo), "bin_high": float(bin_hi),
                    "n": 0, "mean_confidence": float("nan"), "empirical_accuracy": float("nan"),
                })
                continue
            weights = grp_valid["n"].to_numpy(float)
            agg_rows.append({
                "model": model,
                "bin_low": float(bin_lo), "bin_high": float(bin_hi),
                "n": total_n,
                "mean_confidence": float(np.average(grp_valid["mean_confidence"], weights=weights)),
                "empirical_accuracy": float(np.average(grp_valid["empirical_accuracy"], weights=weights)),
            })
        reliability_df = pd.DataFrame(agg_rows).sort_values(["model", "bin_low"]).reset_index(drop=True)
    else:
        reliability_df = pd.DataFrame()

    setup.cfg.output_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(setup.cfg.output_dir / "transition_metrics_per_fold.csv", index=False)
    summary_df.to_csv(setup.cfg.output_dir / "transition_metrics.csv", index=False)
    reliability_df.to_csv(setup.cfg.output_dir / "transition_reliability.csv", index=False)

    return {
        "fold_df": fold_df,
        "summary_df": summary_df,
        "reliability_df": reliability_df,
        "per_visit_logloss": per_visit_logloss,
        "n_eval_visits": int(len(eval_df)),
    }


def _per_row_log_loss(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Per-row negative log probability of the true class."""
    safe = np.clip(probs, 1e-12, 1.0)
    rows = np.arange(len(y_true))
    return -np.log(safe[rows, y_true])


# ---------------------------------------------------------------------------
# eval_dwell
# ---------------------------------------------------------------------------


def eval_dwell(setup: EvalSetup) -> Dict[str, Any]:
    """Predict total dwell length and per-regime distribution shifts by subtype."""
    cfg = setup.cfg
    visits = setup.visits

    if "dwell_time" not in visits.columns or "true_regime_id" not in visits.columns:
        return {"note": "dwell_time or true_regime_id missing — skipping dwell eval"}

    # Build episodes: each is a contiguous run of the same regime for a subject.
    episodes: List[Dict[str, Any]] = []
    for sid, sub in visits.groupby("subject_id"):
        regimes = sub["true_regime_id"].to_numpy(int)
        days = sub["day"].to_numpy(int)
        if len(regimes) == 0:
            continue
        run_start = 0
        for k in range(1, len(regimes)):
            if regimes[k] != regimes[run_start]:
                episodes.append({
                    "subject_id": int(sid),
                    "regime_id": int(regimes[run_start]),
                    "start_day": int(days[run_start]),
                    "end_day": int(days[k - 1]),
                    "length": int(k - run_start),
                    "complete": True,
                })
                run_start = k
        episodes.append({
            "subject_id": int(sid),
            "regime_id": int(regimes[run_start]),
            "start_day": int(days[run_start]),
            "end_day": int(days[-1]),
            "length": int(len(regimes) - run_start),
            "complete": False,
        })
    ep_df = pd.DataFrame(episodes)
    print(f"[dwell] episodes: {len(ep_df)} total, {ep_df['complete'].sum()} complete")

    # Per-regime KS-style mean difference between subtypes.
    subjects = setup.subjects[["subject_id", "hidden_subtype"]]
    merged = ep_df.merge(subjects, on="subject_id", how="left")
    rows: List[Dict[str, Any]] = []
    for regime_id, sub in merged[merged["complete"]].groupby("regime_id"):
        if sub["hidden_subtype"].nunique() < 2:
            continue
        mean_by_subtype = sub.groupby("hidden_subtype")["length"].mean().to_dict()
        rows.append({
            "regime_id": int(regime_id),
            "n_episodes": int(len(sub)),
            "mean_length_overall": float(sub["length"].mean()),
            **{f"mean_length_subtype_{int(k)}": float(v) for k, v in mean_by_subtype.items()},
            "max_minus_min_subtype_mean": float(max(mean_by_subtype.values()) - min(mean_by_subtype.values())),
        })
    dwell_metrics = pd.DataFrame(rows)
    dwell_metrics.to_csv(cfg.output_dir / "dwell_metrics.csv", index=False)
    ep_df.to_csv(cfg.output_dir / "dwell_episodes.csv", index=False)

    return {
        "episodes_df": ep_df,
        "dwell_metrics_df": dwell_metrics,
        "n_episodes": int(len(ep_df)),
        "n_complete": int(ep_df["complete"].sum()),
    }


# ---------------------------------------------------------------------------
# eval_fingerprints
# ---------------------------------------------------------------------------


def _subject_feature_sets(setup: EvalSetup) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Build subject-level feature vectors for several alternative feature sets."""
    visits = setup.visits
    embeddings = setup.embeddings
    embedding_index = setup.embedding_visit_index
    n_states = setup.state_weights.shape[1]
    n_regimes = int(visits["true_regime_id"].max() + 1) if "true_regime_id" in visits.columns else n_states

    # Per-visit embedding lookup keyed by (subject_id, day).
    if {"subject_id", "day"}.issubset(embedding_index.columns):
        embed_df = embedding_index[["subject_id", "day"]].copy()
        for j in range(embeddings.shape[1]):
            embed_df[f"grm_mode_{j + 1}"] = embeddings[:, j]
        v = visits.merge(embed_df, on=["subject_id", "day"], how="left")
    else:
        v = visits.copy()
        mode_columns: List[str] = []
        for j in range(embeddings.shape[1]):
            col = f"grm_mode_{j + 1}"
            mode_columns.append(col)
            v[col] = np.nan
        v.loc[: embeddings.shape[0] - 1, mode_columns] = embeddings

    mode_columns = [c for c in v.columns if c.startswith("grm_mode_")]

    rows: List[Dict[str, Any]] = []
    for sid, sub in v.groupby("subject_id"):
        obs_mean = sub[OBSERVATION_NAMES].mean(numeric_only=True).to_dict()
        regime_occ = np.zeros(n_regimes, dtype=float)
        if "true_regime_id" in sub.columns:
            for r in sub["true_regime_id"].dropna().astype(int):
                regime_occ[int(r)] += 1
            if regime_occ.sum() > 0:
                regime_occ = regime_occ / regime_occ.sum()
        soft_self_res = sub["soft_self_resonance"].dropna()
        sr_stats = {
            "soft_self_res_mean": float(soft_self_res.mean()) if not soft_self_res.empty else 0.0,
            "soft_self_res_std": float(soft_self_res.std(ddof=0)) if not soft_self_res.empty else 0.0,
            "soft_self_res_max": float(soft_self_res.max()) if not soft_self_res.empty else 0.0,
        }
        grm_mean = sub[mode_columns].mean(numeric_only=True).to_dict()
        grm_std = sub[mode_columns].std(numeric_only=True).to_dict()

        per_regime_dwell = {f"mean_dwell_regime_{r}": 0.0 for r in range(n_regimes)}
        if "true_regime_id" in sub.columns and "dwell_time" in sub.columns:
            for r, sr in sub.groupby("true_regime_id"):
                per_regime_dwell[f"mean_dwell_regime_{int(r)}"] = float(sr["dwell_time"].mean())

        row: Dict[str, Any] = {"subject_id": int(sid)}
        for k, val in obs_mean.items():
            row[f"obs_mean_{k}"] = float(val) if pd.notna(val) else 0.0
        for r in range(n_regimes):
            row[f"regime_occ_{r}"] = float(regime_occ[r])
        row.update(sr_stats)
        for k, val in grm_mean.items():
            row[f"{k}_mean"] = float(val) if pd.notna(val) else 0.0
        for k, val in grm_std.items():
            row[f"{k}_std"] = float(val) if pd.notna(val) else 0.0
        row.update(per_regime_dwell)
        rows.append(row)
    fp_df = pd.DataFrame(rows)

    feature_sets: Dict[str, List[str]] = {
        "raw_obs": [c for c in fp_df.columns if c.startswith("obs_mean_")],
        "regime_occupancy": [c for c in fp_df.columns if c.startswith("regime_occ_")],
        "dwell_distribution": [c for c in fp_df.columns if c.startswith("mean_dwell_regime_")],
        "soft_self_resonance": [c for c in fp_df.columns if c.startswith("soft_self_res_")],
        "grm_mean": [c for c in fp_df.columns if c.startswith("grm_mode_") and c.endswith("_mean")],
        "grm_std": [c for c in fp_df.columns if c.startswith("grm_mode_") and c.endswith("_std")],
    }
    feature_sets["strong_baseline"] = feature_sets["raw_obs"] + feature_sets["regime_occupancy"] + feature_sets["dwell_distribution"]
    feature_sets["grm_all"] = feature_sets["grm_mean"] + feature_sets["grm_std"] + feature_sets["soft_self_resonance"]
    feature_sets["strong_plus_grm"] = feature_sets["strong_baseline"] + feature_sets["grm_all"]
    return fp_df, feature_sets


def _cv_macro_f1(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Stratified k-fold logistic regression returning macro-F1 mean, std, per-fold.

    When `groups` is supplied, uses StratifiedGroupKFold so that all rows for a
    given group land in the same fold. This is essential inside a cluster
    bootstrap where the same subject may appear multiple times — without it,
    duplicated rows can leak across folds.
    """
    if groups is not None:
        n_unique = int(len(np.unique(groups)))
        n_splits = min(n_splits, n_unique)
    n_splits = min(n_splits, int(np.min(np.bincount(y))))
    n_splits = max(2, n_splits)

    splitter = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if groups is not None
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    scores: List[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        scaler = StandardScaler()
        split_iter = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
        for tr, te in split_iter:
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(X_tr, y[tr])
            pred = clf.predict(X_te)
            scores.append(float(f1_score(y[te], pred, average="macro", zero_division=0)))
    arr = np.array(scores)
    return float(arr.mean()), float(arr.std(ddof=0)), arr


def eval_fingerprints(setup: EvalSetup) -> Dict[str, Any]:
    """Predict hidden_subtype from several subject feature sets and compare."""
    cfg = setup.cfg
    fp_df, feature_sets = _subject_feature_sets(setup)
    target = setup.subjects[["subject_id", "hidden_subtype"]]
    merged = fp_df.merge(target, on="subject_id", how="inner").reset_index(drop=True)
    y = merged["hidden_subtype"].to_numpy(int)

    rows: List[Dict[str, Any]] = []
    for name, cols in feature_sets.items():
        cols = [c for c in cols if c in merged.columns]
        if not cols:
            rows.append({"feature_set": name, "n_features": 0, "macro_f1_mean": float("nan"), "macro_f1_std": float("nan")})
            continue
        X = merged[cols].to_numpy(float)
        X = SimpleImputer(strategy="median").fit_transform(X)
        mean, std, per_fold = _cv_macro_f1(X, y, cfg.cv_splits, cfg.seed)
        rows.append({
            "feature_set": name,
            "n_features": len(cols),
            "macro_f1_mean": mean,
            "macro_f1_std": std,
            "per_fold": json.dumps(per_fold.round(4).tolist()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.output_dir / "subject_fingerprint_metrics.csv", index=False)
    fp_df.to_csv(cfg.output_dir / "subject_fingerprints.csv", index=False)

    strong = float(summary[summary["feature_set"] == "strong_baseline"]["macro_f1_mean"].iloc[0])
    combined = float(summary[summary["feature_set"] == "strong_plus_grm"]["macro_f1_mean"].iloc[0])
    grm_lift = combined - strong

    # Bootstrap the lift over subjects (each subject is one row here, so no
    # within-subject clustering — straight subject-level resampling).
    strong_cols = [c for c in feature_sets["strong_baseline"] if c in merged.columns]
    combo_cols = [c for c in feature_sets["strong_plus_grm"] if c in merged.columns]
    X_strong = SimpleImputer(strategy="median").fit_transform(merged[strong_cols].to_numpy(float))
    X_combo = SimpleImputer(strategy="median").fit_transform(merged[combo_cols].to_numpy(float))

    subj_ids = merged["subject_id"].to_numpy(int)

    def _fp_paired(idx: np.ndarray) -> Tuple[float, float, float]:
        if idx.size < 6 or len(np.unique(y[idx])) < 2:
            return (float("nan"), float("nan"), float("nan"))
        g = subj_ids[idx]
        if len(np.unique(g)) < cfg.cv_splits:
            return (float("nan"), float("nan"), float("nan"))
        try:
            f_strong, _, _ = _cv_macro_f1(X_strong[idx], y[idx], cfg.cv_splits, cfg.seed, groups=g)
            f_combo, _, _ = _cv_macro_f1(X_combo[idx], y[idx], cfg.cv_splits, cfg.seed, groups=g)
            return (f_strong, f_combo, f_combo - f_strong)
        except Exception:
            return (float("nan"), float("nan"), float("nan"))

    fp_results = cluster_bootstrap_paired(
        subj_ids, _fp_paired, n_boot=cfg.bootstrap_n, seed=cfg.seed, label="fingerprint",
    )
    (strong_pt, strong_lo, strong_hi) = fp_results[0]
    (combo_pt, combo_lo, combo_hi) = fp_results[1]
    (lift_pt, lift_lo, lift_hi) = fp_results[2]

    return {
        "summary_df": summary,
        "fingerprint_df": merged,
        "strong_baseline_f1": strong,
        "strong_plus_grm_f1": combined,
        "grm_lift_over_strong": grm_lift,
        "strong_baseline_ci": (strong_pt, strong_lo, strong_hi),
        "strong_plus_grm_ci": (combo_pt, combo_lo, combo_hi),
        "grm_lift_ci": (lift_pt, lift_lo, lift_hi),
    }


# ---------------------------------------------------------------------------
# eval_aliased
# ---------------------------------------------------------------------------


def _build_aliased_mask(setup: EvalSetup) -> np.ndarray:
    """Visit i is aliased if its observation-NN set has both attractor and non-attractor members."""
    visits = setup.visits
    obs = visits[OBSERVATION_NAMES].copy()
    obs = SimpleImputer(strategy="median").fit_transform(obs)
    X = StandardScaler().fit_transform(obs)
    k = setup.cfg.aliased_k_nn + 1
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]  # drop self
    attractor = visits["attractor_state"].to_numpy(int) if "attractor_state" in visits.columns else np.zeros(len(visits), dtype=int)
    mins = setup.cfg.aliased_min_each
    n_attr = attractor[idx].sum(axis=1)
    n_non = setup.cfg.aliased_k_nn - n_attr
    return (n_attr >= mins) & (n_non >= mins)


def _knn_score_attr(setup: EvalSetup, X: np.ndarray, k: int) -> np.ndarray:
    """For each row, mean attractor_state among k observation neighbors (excluding self)."""
    visits = setup.visits
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]
    attractor = visits["attractor_state"].to_numpy(int) if "attractor_state" in visits.columns else np.zeros(len(visits), dtype=int)
    return attractor[idx].mean(axis=1)


def _neighborhood_entropy(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Per-row Shannon entropy of the label distribution in its k-NN neighborhood."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]
    out = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        vals, counts = np.unique(labels[idx[i]], return_counts=True)
        p = counts / counts.sum()
        out[i] = float(-(p * np.log(p + 1e-12)).sum())
    return out


def eval_aliased(setup: EvalSetup) -> Dict[str, Any]:
    """T1-T4 aliased-state separability tests."""
    cfg = setup.cfg
    visits = setup.visits
    n_visits = len(visits)
    aliased_mask = _build_aliased_mask(setup)
    n_aliased = int(aliased_mask.sum())
    print(f"[aliased] {n_aliased} aliased visits out of {n_visits}")

    obs_features = visits[OBSERVATION_NAMES].copy()
    obs_features = SimpleImputer(strategy="median").fit_transform(obs_features)
    obs_features = StandardScaler().fit_transform(obs_features)

    embed_df = setup.embedding_visit_index[["subject_id", "day"]].copy() if "subject_id" in setup.embedding_visit_index.columns else None
    if embed_df is not None:
        for j in range(setup.embeddings.shape[1]):
            embed_df[f"m{j}"] = setup.embeddings[:, j]
        v = visits[["subject_id", "day"]].merge(embed_df, on=["subject_id", "day"], how="left")
        mode_cols = [c for c in v.columns if c.startswith("m") and len(c) <= 4]
        grm_features = v[mode_cols].to_numpy(float)
    else:
        grm_features = np.zeros((n_visits, 1))
    grm_features = SimpleImputer(strategy="median").fit_transform(grm_features)
    grm_features = StandardScaler().fit_transform(grm_features)

    regimes = visits["true_regime_id"].to_numpy(int) if "true_regime_id" in visits.columns else np.zeros(n_visits, dtype=int)
    next_regimes = visits["next_true_regime_id"]

    # T1: neighborhood entropy in obs vs GRM space, on aliased subset.
    ent_obs = _neighborhood_entropy(obs_features, regimes, cfg.aliased_k_nn)
    ent_grm = _neighborhood_entropy(grm_features, regimes, cfg.aliased_k_nn)
    win_grm = (ent_grm[aliased_mask] < ent_obs[aliased_mask])
    subj_full = visits["subject_id"].to_numpy(int)
    aliased_subj = subj_full[aliased_mask]
    t1_winrate, t1_lo, t1_hi = bootstrap_ci(
        win_grm.astype(float), groups=aliased_subj, n_boot=cfg.bootstrap_n, seed=cfg.seed,
    )

    # T2: AUC for attractor_state among aliased visits with cluster bootstrap.
    attractor = visits["attractor_state"].to_numpy(int) if "attractor_state" in visits.columns else np.zeros(n_visits, dtype=int)
    score_obs = _knn_score_attr(setup, obs_features, cfg.aliased_k_nn)
    score_grm = _knn_score_attr(setup, grm_features, cfg.aliased_k_nn)
    aliased_idx_full = np.where(aliased_mask)[0]

    aliased_set = set(aliased_idx_full.tolist())

    def _t2(idx: np.ndarray) -> Tuple[float, float, float]:
        # np.isin preserves duplicates — a visit whose subject is sampled twice
        # appears twice in `keep`, which is the correct cluster-bootstrap weight.
        keep = idx[np.isin(idx, aliased_idx_full)]
        if keep.size < 5:
            return (float("nan"), float("nan"), float("nan"))
        y = attractor[keep]
        if y.sum() == 0 or y.sum() == len(y):
            return (float("nan"), float("nan"), float("nan"))
        a_obs = float(roc_auc_score(y, score_obs[keep]))
        a_grm = float(roc_auc_score(y, score_grm[keep]))
        return (a_obs, a_grm, a_grm - a_obs)

    t2_results = cluster_bootstrap_paired(
        subj_full, _t2, n_boot=cfg.bootstrap_n, seed=cfg.seed, label="T2_attractor_auc",
    )
    (t2_obs_auc, t2_obs_lo, t2_obs_hi) = t2_results[0]
    (t2_grm_auc, t2_grm_lo, t2_grm_hi) = t2_results[1]
    (t2_lift, t2_lift_lo, t2_lift_hi) = t2_results[2]

    # T3: per-row CV predictions on aliased subset, then cluster bootstrap.
    mask3 = aliased_mask & next_regimes.notna().to_numpy(bool)
    y3 = next_regimes[mask3].to_numpy(int)
    groups3 = subj_full[mask3]
    pred_obs3, pred_grm3 = _aliased_next_regime_predictions(
        obs_features[mask3], grm_features[mask3], y3, groups3, cfg,
    )
    # Rows that GroupKFold didn't assign to any test fold (degenerate fold-count
    # or single-group fold) remain at -1; exclude them from accuracy.
    valid3 = (pred_obs3 >= 0) & (pred_grm3 >= 0)
    if valid3.sum() < len(valid3):
        print(f"[T3] dropping {int((~valid3).sum())}/{len(valid3)} aliased rows with no CV prediction")
    correct_obs3 = (pred_obs3 == y3).astype(float)
    correct_grm3 = (pred_grm3 == y3).astype(float)

    def _t3(local_idx: np.ndarray) -> Tuple[float, float, float]:
        if local_idx.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        keep = local_idx[valid3[local_idx]]
        if keep.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        a_obs = float(correct_obs3[keep].mean())
        a_grm = float(correct_grm3[keep].mean())
        return (a_obs, a_grm, a_grm - a_obs)

    t3_results = cluster_bootstrap_paired(
        groups3, _t3, n_boot=cfg.bootstrap_n, seed=cfg.seed, label="T3_next_regime_top1",
    )
    (t3_obs_acc, t3_obs_lo, t3_obs_hi) = t3_results[0]
    (t3_grm_acc, t3_grm_lo, t3_grm_hi) = t3_results[1]
    (t3_lift, t3_lift_lo, t3_lift_hi) = t3_results[2]

    # T4: silhouette by true_regime, cluster bootstrap (capped at 80 reps; silhouette is O(n^2)).
    t4_boot = min(cfg.bootstrap_n, 80)

    def _t4(idx: np.ndarray) -> Tuple[float, float, float]:
        # Preserve duplicates: see _t2.
        keep = idx[np.isin(idx, aliased_idx_full)]
        if keep.size < 10:
            return (float("nan"), float("nan"), float("nan"))
        regs = regimes[keep]
        if len(np.unique(regs)) < 2:
            return (float("nan"), float("nan"), float("nan"))
        s_obs = _safe_silhouette(obs_features[keep], regs)
        s_grm = _safe_silhouette(grm_features[keep], regs)
        return (s_obs, s_grm, s_grm - s_obs)

    t4_results = cluster_bootstrap_paired(
        subj_full, _t4, n_boot=t4_boot, seed=cfg.seed, label="T4_silhouette",
    )
    (sil_obs, sil_obs_lo, sil_obs_hi) = t4_results[0]
    (sil_grm, sil_grm_lo, sil_grm_hi) = t4_results[1]
    (sil_lift, sil_lift_lo, sil_lift_hi) = t4_results[2]

    def _row(test: str, val: float, lo: float, hi: float, n: int) -> Dict[str, Any]:
        return {"test": test, "value": val, "ci_low": lo, "ci_high": hi, "n": int(n)}

    n_alias = int(aliased_mask.sum())
    summary = pd.DataFrame([
        _row("T1_neighborhood_entropy_winrate_grm", t1_winrate, t1_lo, t1_hi, n_alias),
        _row("T2_attractor_auc_obs", t2_obs_auc, t2_obs_lo, t2_obs_hi, n_alias),
        _row("T2_attractor_auc_grm", t2_grm_auc, t2_grm_lo, t2_grm_hi, n_alias),
        _row("T2_attractor_auc_lift_grm_minus_obs", t2_lift, t2_lift_lo, t2_lift_hi, n_alias),
        _row("T3_next_regime_top1_obs", t3_obs_acc, t3_obs_lo, t3_obs_hi, int(mask3.sum())),
        _row("T3_next_regime_top1_grm", t3_grm_acc, t3_grm_lo, t3_grm_hi, int(mask3.sum())),
        _row("T3_next_regime_top1_lift_grm_minus_obs", t3_lift, t3_lift_lo, t3_lift_hi, int(mask3.sum())),
        _row("T4_silhouette_obs", sil_obs, sil_obs_lo, sil_obs_hi, n_alias),
        _row("T4_silhouette_grm", sil_grm, sil_grm_lo, sil_grm_hi, n_alias),
        _row("T4_silhouette_lift_grm_minus_obs", sil_lift, sil_lift_lo, sil_lift_hi, n_alias),
    ])
    summary.to_csv(cfg.output_dir / "aliased_state_analysis.csv", index=False)
    return {
        "summary_df": summary,
        "aliased_mask": aliased_mask,
        "t1_winrate": (t1_winrate, t1_lo, t1_hi),
        "t2_obs_auc": (t2_obs_auc, t2_obs_lo, t2_obs_hi),
        "t2_grm_auc": (t2_grm_auc, t2_grm_lo, t2_grm_hi),
        "t2_lift": (t2_lift, t2_lift_lo, t2_lift_hi),
        "t3_obs_acc": (t3_obs_acc, t3_obs_lo, t3_obs_hi),
        "t3_grm_acc": (t3_grm_acc, t3_grm_lo, t3_grm_hi),
        "t3_lift": (t3_lift, t3_lift_lo, t3_lift_hi),
        "t4_sil_obs": (sil_obs, sil_obs_lo, sil_obs_hi),
        "t4_sil_grm": (sil_grm, sil_grm_lo, sil_grm_hi),
        "t4_sil_lift": (sil_lift, sil_lift_lo, sil_lift_hi),
    }


def _aliased_next_regime_predictions(
    X_obs: np.ndarray, X_grm: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: DynamicEvalConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subject-grouped CV; return per-row predicted class for obs and GRM features."""
    n = len(y)
    pred_obs = np.full(n, -1, dtype=int)
    pred_grm = np.full(n, -1, dtype=int)
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return pred_obs, pred_grm
    n_splits = min(cfg.cv_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        for tr, te in gkf.split(X_obs, y, groups=groups):
            for X, preds in [(X_obs, pred_obs), (X_grm, pred_grm)]:
                clf = LogisticRegression(max_iter=2000, C=1.0, random_state=cfg.seed)
                clf.fit(X[tr], y[tr])
                preds[te] = clf.predict(X[te])
    return pred_obs, pred_grm


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score with NaN guard for tiny or single-class inputs."""
    if len(X) < 3 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# eval_ablations
# ---------------------------------------------------------------------------


def eval_ablations(setup: EvalSetup, aliased_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Re-run aliased T2 on perturbed embeddings; compare to full GRM."""
    cfg = setup.cfg
    visits = setup.visits
    n_visits = len(visits)
    embeddings = setup.embeddings

    if "attractor_state" not in visits.columns or "true_regime_id" not in visits.columns:
        return {"note": "missing attractor_state or true_regime_id — skipping ablations"}

    embed_df = setup.embedding_visit_index[["subject_id", "day"]].copy()
    for j in range(embeddings.shape[1]):
        embed_df[f"m{j}"] = embeddings[:, j]
    v = visits[["subject_id", "day"]].merge(embed_df, on=["subject_id", "day"], how="left")
    mode_cols = [c for c in v.columns if c.startswith("m") and len(c) <= 4]
    grm_full = v[mode_cols].to_numpy(float)

    rng = np.random.default_rng(cfg.seed + 17)

    # Shuffled time within subject: permute embedding-row → day mapping within each subject.
    grm_shuffled_time = grm_full.copy()
    subj = v["subject_id"].to_numpy(int)
    for sid in np.unique(subj):
        idx = np.where(subj == sid)[0]
        perm = rng.permutation(idx)
        grm_shuffled_time[idx] = grm_full[perm]

    # Random embedding control: same shape, drawn from standard normal.
    grm_random = rng.normal(size=grm_full.shape)

    # Raw observation control.
    obs = SimpleImputer(strategy="median").fit_transform(visits[OBSERVATION_NAMES])
    obs = StandardScaler().fit_transform(obs)

    aliased_mask = aliased_result["aliased_mask"] if aliased_result else _build_aliased_mask(setup)
    attractor = visits["attractor_state"].to_numpy(int)
    y_aliased = attractor[aliased_mask]
    if y_aliased.sum() == 0 or y_aliased.sum() == len(y_aliased):
        return {"note": "aliased subset has single attractor class — skipping ablations"}

    def t2_auc(X: np.ndarray) -> float:
        X_imp = SimpleImputer(strategy="median").fit_transform(X)
        X_sc = StandardScaler().fit_transform(X_imp)
        score = _knn_score_attr(setup, X_sc, cfg.aliased_k_nn)
        return float(roc_auc_score(y_aliased, score[aliased_mask]))

    rows = []
    for name, X in [
        ("full_grm", grm_full),
        ("shuffled_time_grm", grm_shuffled_time),
        ("random_embedding", grm_random),
        ("raw_observations", obs),
    ]:
        rows.append({"ablation": name, "t2_attractor_auc_aliased": t2_auc(X)})

    df = pd.DataFrame(rows)
    df.to_csv(cfg.output_dir / "ablation_metrics.csv", index=False)
    return {"df": df}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    save_with_caption(fig, path, dpi=140)


def generate_plots(setup: EvalSetup, results: Dict[str, Any]) -> None:
    """Render the headline plots."""
    plot_dir = setup.cfg.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Transition log-loss bar chart.
    if "transitions" in results:
        summary = results["transitions"]["summary_df"]
        ll = summary[summary["metric"] == "log_loss"].copy()
        if not ll.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            y = ll["mean"].to_numpy()
            err_low = ll["mean"].to_numpy() - ll["ci_low"].to_numpy()
            err_high = ll["ci_high"].to_numpy() - ll["mean"].to_numpy()
            ax.bar(ll["model"], y, yerr=[err_low, err_high], capsize=4)
            ax.set_ylabel("Log-loss (lower is better)")
            ax.set_title("Transition log-loss by model (subject-CV, bootstrap 95% CI)")
            ax.tick_params(axis="x", rotation=20)
            _save(fig, plot_dir / "transition_log_loss_by_model.png")

        # Reliability diagram.
        rel = results["transitions"]["reliability_df"]
        if not rel.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "--", linewidth=1)
            for model, sub in rel.groupby("model"):
                ax.plot(sub["mean_confidence"], sub["empirical_accuracy"], marker="o", label=model)
            ax.set_xlabel("Mean predicted top-1 probability")
            ax.set_ylabel("Empirical top-1 accuracy")
            ax.set_title("Reliability diagram (subject-CV folds, fold-weighted)")
            ax.legend(fontsize=8, loc="best")
            _save(fig, plot_dir / "transition_reliability.png")

    # Subject fingerprint comparison.
    if "fingerprints" in results:
        summary = results["fingerprints"]["summary_df"]
        if not summary.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(summary["feature_set"], summary["macro_f1_mean"], yerr=summary["macro_f1_std"], capsize=4)
            ax.set_ylabel("Hidden_subtype macro-F1 (5-fold stratified)")
            ax.set_title("Subject fingerprint recovery by feature set")
            ax.tick_params(axis="x", rotation=30)
            _save(fig, plot_dir / "subject_fingerprint_macro_f1.png")

    # Aliased-state scatter (raw PCA vs first 2 GRM modes, aliased subset).
    if "aliased" in results and "aliased_mask" in results["aliased"]:
        aliased = results["aliased"]["aliased_mask"]
        visits = setup.visits
        embed_df = setup.embedding_visit_index[["subject_id", "day"]].copy()
        for j in range(setup.embeddings.shape[1]):
            embed_df[f"m{j}"] = setup.embeddings[:, j]
        v = visits[["subject_id", "day", "true_regime_id"]].merge(embed_df, on=["subject_id", "day"], how="left")
        obs = SimpleImputer(strategy="median").fit_transform(visits[OBSERVATION_NAMES])
        obs = StandardScaler().fit_transform(obs)
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2).fit_transform(obs)
        regimes = v["true_regime_id"].fillna(-1).astype(int).to_numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax, X, title in [
            (axes[0], pca, "Raw observations (PCA-2)"),
            (axes[1], v[["m0", "m1"]].to_numpy(float), "GRM modes 1-2"),
        ]:
            mask_known = aliased & (regimes >= 0) & np.all(np.isfinite(X), axis=1)
            for r in np.unique(regimes[mask_known]):
                m = mask_known & (regimes == r)
                ax.scatter(X[m, 0], X[m, 1], s=10, alpha=0.55, label=f"r{int(r)}")
            ax.set_title(title)
            ax.legend(fontsize=7, loc="best", markerscale=1.2)
        fig.suptitle("Aliased visits: observation PCA vs GRM modes, by true_regime")
        _save(fig, plot_dir / "aliased_visits_scatter.png")

    # Ablation bar chart.
    if "ablations" in results and isinstance(results["ablations"], dict) and "df" in results["ablations"]:
        df = results["ablations"]["df"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(df["ablation"], df["t2_attractor_auc_aliased"])
        ax.set_ylabel("Attractor AUC on aliased subset")
        ax.set_title("T2 attractor AUC by embedding source (aliased visits only)")
        ax.tick_params(axis="x", rotation=15)
        ax.axhline(0.5, linestyle="--", linewidth=0.8)
        _save(fig, plot_dir / "ablation_attractor_auc.png")


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


def write_certificate(setup: EvalSetup, results: Dict[str, Any]) -> Dict[str, Any]:
    """Write a structured falsifiable-verdicts JSON summarizing all eval results."""
    cert: Dict[str, Any] = {
        "config": asdict(setup.cfg) | {"data_dir": str(setup.cfg.data_dir),
                                        "static_model_dir": str(setup.cfg.static_model_dir),
                                        "dynamic_model_dir": str(setup.cfg.dynamic_model_dir),
                                        "output_dir": str(setup.cfg.output_dir)},
        "verdicts": {},
        "framing": (
            "Diagnostic on a known synthetic generator. Not a biological simulator. Not evidence for "
            "TCM or Qi. Verdicts describe latent-state recovery, attractor fingerprinting, and "
            "ontology-mismatch detection on synthetic data."
        ),
        "caveats": {
            "grm_matrices_not_refit_per_fold": (
                "The persisted GRM-blended transition matrices G^(t) were fit on all training "
                "subjects, not refit per CV fold. Evaluation is subject-CV honest at the prediction "
                "layer but the GRM model itself has seen all subjects. This is leakage in GRM's "
                "favor; any FAIL verdict against GRM is therefore conservative."
            ),
            "strong_baseline_uses_state_id": (
                "Strong baseline F uses state_id (cluster vocabulary), matching what GRM has access "
                "to. Earlier versions used true_regime_id, which gave the baseline oracle features "
                "GRM did not have."
            ),
            "fingerprint_bootstrap_uses_grouped_cv": (
                "Inside the fingerprint cluster bootstrap, _cv_macro_f1 uses StratifiedGroupKFold "
                "by subject_id so duplicated subjects (a normal cluster-bootstrap outcome) cannot "
                "leak across train/test folds."
            ),
            "aliased_bootstrap_preserves_duplicates": (
                "T2 and T4 cluster bootstrap use np.isin (not np.intersect1d) so a visit whose "
                "subject was sampled k times contributes k copies — the correct cluster-bootstrap "
                "weight for AUC and silhouette."
            ),
        },
    }

    if "transitions" in results:
        summary = results["transitions"]["summary_df"]
        ll = summary[summary["metric"] == "log_loss"].set_index("model")
        if {"A_markov", "C_grm_blended", "F_mnl_baseline"}.issubset(ll.index):
            cert["verdicts"]["grm_beats_markov_log_loss"] = _verdict(
                ll.loc["A_markov", "mean"] - ll.loc["C_grm_blended", "mean"],
                ll.loc["A_markov", "ci_low"] - ll.loc["C_grm_blended", "ci_high"],
                ll.loc["A_markov", "ci_high"] - ll.loc["C_grm_blended", "ci_low"],
                positive_is_pass=True,
                description="GRM-blended transition log-loss vs empirical Markov (positive = GRM helps).",
            )
            cert["verdicts"]["grm_beats_strong_baseline_log_loss"] = _verdict(
                ll.loc["F_mnl_baseline", "mean"] - ll.loc["C_grm_blended", "mean"],
                ll.loc["F_mnl_baseline", "ci_low"] - ll.loc["C_grm_blended", "ci_high"],
                ll.loc["F_mnl_baseline", "ci_high"] - ll.loc["C_grm_blended", "ci_low"],
                positive_is_pass=True,
                description="GRM-blended vs strong baseline (current_regime + dwell + delayed_loads).",
            )
        brier = summary[summary["metric"] == "brier"].set_index("model")
        if {"A_markov", "C_grm_blended"}.issubset(brier.index):
            cert["verdicts"]["grm_beats_markov_brier"] = _verdict(
                brier.loc["A_markov", "mean"] - brier.loc["C_grm_blended", "mean"],
                brier.loc["A_markov", "ci_low"] - brier.loc["C_grm_blended", "ci_high"],
                brier.loc["A_markov", "ci_high"] - brier.loc["C_grm_blended", "ci_low"],
                positive_is_pass=True,
                description="Brier score: GRM-blended vs empirical Markov.",
            )

    if "aliased" in results:
        a = results["aliased"]
        t1_pt, t1_lo, t1_hi = a["t1_winrate"]
        cert["verdicts"]["grm_separates_aliased_states"] = _verdict(
            t1_pt - 0.5, t1_lo - 0.5, t1_hi - 0.5,
            positive_is_pass=True,
            description="Fraction of aliased visits where GRM neighborhood entropy < observation entropy (positive = GRM separates).",
        )
        t2_lift_pt, t2_lift_lo, t2_lift_hi = a["t2_lift"]
        cert["verdicts"]["grm_attractor_auc_lift_aliased"] = _verdict(
            t2_lift_pt, t2_lift_lo, t2_lift_hi,
            positive_is_pass=True,
            description="AUC lift for attractor_state via k-NN on aliased visits: GRM minus obs (subject cluster bootstrap).",
        )
        t3_lift_pt, t3_lift_lo, t3_lift_hi = a["t3_lift"]
        cert["verdicts"]["grm_next_regime_top1_lift_aliased"] = _verdict(
            t3_lift_pt, t3_lift_lo, t3_lift_hi,
            positive_is_pass=True,
            description="Top-1 next_regime accuracy lift on aliased visits: GRM minus obs (subject cluster bootstrap).",
        )
        t4_lift_pt, t4_lift_lo, t4_lift_hi = a["t4_sil_lift"]
        cert["verdicts"]["grm_silhouette_lift_aliased"] = _verdict(
            t4_lift_pt, t4_lift_lo, t4_lift_hi,
            positive_is_pass=True,
            description="Silhouette lift by true_regime on aliased visits: GRM minus obs (subject cluster bootstrap).",
        )

    if "fingerprints" in results:
        f = results["fingerprints"]
        lift_pt, lift_lo, lift_hi = f.get("grm_lift_ci", (f["grm_lift_over_strong"], float("nan"), float("nan")))
        cert["verdicts"]["grm_helps_hidden_subtype_recovery"] = _verdict(
            lift_pt, lift_lo, lift_hi,
            positive_is_pass=True,
            description="Hidden_subtype macro-F1 lift: (strong_baseline + GRM) minus strong_baseline (subject bootstrap).",
        )
        cert["verdicts"]["grm_helps_hidden_subtype_recovery"]["strong_baseline_f1"] = f["strong_baseline_f1"]
        cert["verdicts"]["grm_helps_hidden_subtype_recovery"]["strong_plus_grm_f1"] = f["strong_plus_grm_f1"]

    if "ablations" in results and "df" in results["ablations"]:
        df = results["ablations"]["df"]
        full = float(df[df["ablation"] == "full_grm"]["t2_attractor_auc_aliased"].iloc[0]) if not df.empty else float("nan")
        for name in ["shuffled_time_grm", "random_embedding", "raw_observations"]:
            sub = df[df["ablation"] == name]
            if sub.empty:
                continue
            val = float(sub["t2_attractor_auc_aliased"].iloc[0])
            cert.setdefault("ablations", {})[name] = {"t2_attractor_auc_aliased": val, "vs_full_grm": full - val}

    path = setup.cfg.output_dir / "dynamic_eval_certificates.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2, default=str)
    return cert


def _verdict(magnitude: float, ci_low: float, ci_high: float, *, positive_is_pass: bool, description: str) -> Dict[str, Any]:
    """Wrap a metric + CI into a structured verdict."""
    if positive_is_pass:
        if np.isfinite(ci_low) and ci_low > 0:
            passes = True
        elif np.isfinite(ci_high) and ci_high < 0:
            passes = False
        else:
            passes = None
    else:
        if np.isfinite(ci_high) and ci_high < 0:
            passes = True
        elif np.isfinite(ci_low) and ci_low > 0:
            passes = False
        else:
            passes = None
    return {
        "magnitude": float(magnitude),
        "ci_low": float(ci_low) if np.isfinite(ci_low) else None,
        "ci_high": float(ci_high) if np.isfinite(ci_high) else None,
        "passes": passes,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------


def print_summary(cert: Dict[str, Any], setup: EvalSetup) -> None:
    """Print headline verdicts at the end of the run."""
    print()
    print("=" * 72)
    print("Dynamic eval — headline verdicts")
    print("=" * 72)
    print(f"Outputs written to: {setup.cfg.output_dir.resolve()}")
    print()
    verdicts = cert.get("verdicts", {})
    for key, v in verdicts.items():
        if not isinstance(v, dict):
            continue
        mag = v.get("magnitude")
        ci_lo = v.get("ci_low")
        ci_hi = v.get("ci_high")
        passes = v.get("passes")
        verdict_str = (
            "PASS" if passes is True else "FAIL" if passes is False else "MARGINAL"
        )
        if ci_lo is not None and ci_hi is not None:
            ci_str = f"({ci_lo:+.4f}, {ci_hi:+.4f})"
        else:
            ci_str = ""
        print(f"  [{verdict_str:8s}] {key:48s} Δ={mag:+.4f} {ci_str}")
        print(f"             {v.get('description', '')}")
    print()
    print("Scientific framing: " + cert.get("framing", ""))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all selected eval modules in order, writing outputs and a certificate."""
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    scope = set(cfg.scope)
    if "all" in scope:
        scope = {"transitions", "dwell", "fingerprints", "aliased", "ablations", "plots"}

    setup = build_setup(cfg)
    results: Dict[str, Any] = {}

    if "transitions" in scope:
        print("[1] eval_transitions")
        results["transitions"] = eval_transitions(setup)
    if "dwell" in scope:
        print("[2] eval_dwell")
        results["dwell"] = eval_dwell(setup)
    if "fingerprints" in scope:
        print("[3] eval_fingerprints")
        results["fingerprints"] = eval_fingerprints(setup)
    if "aliased" in scope:
        print("[4] eval_aliased")
        results["aliased"] = eval_aliased(setup)
    if "ablations" in scope:
        print("[5] eval_ablations")
        results["ablations"] = eval_ablations(setup, results.get("aliased"))
    if "plots" in scope:
        print("[6] generate_plots")
        generate_plots(setup, results)

    cert = write_certificate(setup, results)
    print_summary(cert, setup)


if __name__ == "__main__":
    main()
