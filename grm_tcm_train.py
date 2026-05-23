from __future__ import annotations

"""
GRM-TCM trainer / evaluator for the synthetic dataset.

Run sequence:
  python grm_tcm_synthetic_generator.py
  python grm_tcm_train.py

Expected input:
  synthetic_grm_tcm/visits.csv
  synthetic_grm_tcm/latent_states.csv
  synthetic_grm_tcm/events.csv

Outputs:
  grm_tcm_results/grm_visit_embeddings.csv
  grm_tcm_results/grm_feature_modes.csv
  grm_tcm_results/grm_predictions.csv
  grm_tcm_results/grm_metrics.json
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigsh
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grm_tcm_persistence import (
    canonicalize_eigvec_signs,
    save_joblib,
    write_manifest,
)
from grm_tcm_projection import nystrom_extend_arrays, surrogate_project


STATIC_SCHEMA_VERSION = "static-v3"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for arbitrary real-valued inputs."""

    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _fit_binary_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Find T > 0 minimizing NLL of sigmoid(z / T) against binary labels y."""

    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=int)
    if z.size == 0 or len(np.unique(y)) < 2:
        return 1.0

    def nll(t: float) -> float:
        t_safe = max(float(t), 1e-3)
        s = z / t_safe
        # log(sigmoid(s)) and log(1 - sigmoid(s)) via softplus identities
        log_p1 = -np.logaddexp(0.0, -s)
        log_p0 = -np.logaddexp(0.0, s)
        return float(-np.mean(np.where(y == 1, log_p1, log_p0)))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


OBSERVATION_NAMES = [
    "sleep_quality", "hrv", "resting_hr", "body_temp", "fatigue", "pain",
    "appetite", "bowel_quality", "mood_calm", "energy", "heaviness", "cold_hot",
]

# Ordinal observations from the v2 generator (pulse / tongue / complexion-like).
# Stored in visits.csv as integer levels; included as ordinal-as-continuous inputs
# alongside the 12 continuous channels when present and config flag is on.
QUALITATIVE_FEATURE_NAMES = [
    "pulse_quality_like", "tongue_state_like", "complexion_like",
]

# Stable per-subject constitution axes (v2 generator). Used as targets for the
# constitution-recovery evaluation, not as input features (they live in subjects.csv).
CONSTITUTION_NAMES = [
    "constitution_thermal", "constitution_energy", "constitution_stability",
]

LATENT_NAMES = [
    "vitality_depletion", "stress_activation", "inflammatory_load", "digestive_instability",
]


@dataclass
class GRMTrainConfig:
    input_dir: str = "synthetic_grm_tcm"
    output_dir: str = "grm_tcm_results"
    random_seed: int = 42

    n_neighbors: int = 12
    similarity_sigma: Optional[float] = None
    diffusion_alpha: float = 1.0
    temporal_edge_weight: float = 0.75
    same_subject_edge_weight: float = 0.15
    treatment_edge_weight: float = 0.20
    subject_similarity_edge_weight: float = 0.25
    subject_similarity_neighbors: int = 4
    graph_mode: str = "feature_temporal_treatment"

    n_modes: int = 8
    rho: float = 1.0
    use_normalized_laplacian: bool = True

    test_size: float = 0.25
    target_regression: str = "next_day_score"
    target_classification: str = "flare_next_day"
    # Secondary classification target: flare ONSET (today=0 -> tomorrow=1).
    # Persistence baseline collapses here (it predicts 0 for all flare_today=0 rows),
    # so GRM/embedding-based heads should beat it cleanly when there is real signal.
    target_classification_onset: str = "flare_onset"
    # If True, the GRM heads also see the per-subject lag of the targets
    # (score_persistence_today, flare_persistence_today) as additional features.
    # This makes the headline "grm" head a fair deployment-style competitor
    # against the pure-persistence baseline rather than a strawman.
    use_lag_features: bool = True
    # If True, include v2 qualitative ordinal channels (pulse/tongue/complexion-like)
    # in the observation matrix when present. The wider feature set is also used
    # by the raw-RF baseline so the comparison stays apples-to-apples.
    include_qualitative_features: bool = True
    # Delay-embedding window size for Takens baseline (number of consecutive visits
    # concatenated into a single feature vector). Set to 1 to disable (snapshot only).
    # Respects subject boundaries; early visits padded with NaN then median-imputed.
    delay_embedding_k: int = 3
    # If True, run constitution-recovery evaluation in inductive AND transductive
    # modes. Reports visit-GRM aggregates, raw subject aggregates, and a dedicated
    # subject-level GRM diagnostic so stable constitution is not forced through a
    # visit-only spectral geometry. Skipped silently if subjects.csv lacks
    # constitution columns.
    evaluate_constitution_recovery: bool = True

    # Strict inductive evaluation: split subjects first, fit scaler/KNN/graph/
    # eigenbasis ONLY on train subjects, then project test subjects via the
    # chosen projection method ('surrogate' or 'nystrom'). Reports honest
    # held-out metrics; the persisted model is the train-only fit.
    inductive: bool = False
    projection: str = "surrogate"
    n_neighbors_inductive: int = 12
    # Where to look for the matching transductive metrics when generating the
    # transductive-vs-inductive comparison plot in inductive mode. Default
    # assumes the standard layout: sibling `grm_tcm_results/` dir.
    transductive_results_dir: str = "grm_tcm_results"


class GRMTCMTrainer:
    def __init__(self, config: GRMTrainConfig):
        self.cfg = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.obs_preprocessor: Optional[Pipeline] = None
        self.nn_index: Optional[NearestNeighbors] = None
        self.knn_sigma: Optional[float] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.eigenvalues_full: Optional[np.ndarray] = None
        self.train_degrees: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.ridge_reg: Optional[BaseEstimator] = None
        self.logistic_clf: Optional[BaseEstimator] = None
        self.embedding_surrogate: Optional[BaseEstimator] = None
        self.flare_temperature: Optional[float] = None
        self.train_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None
        self.procrustes_R: Optional[np.ndarray] = None
        self._visit_index: Optional[pd.DataFrame] = None
        self._X_obs: Optional[np.ndarray] = None

    def run(self) -> Dict:
        visits, latent, events = self._load_data()
        visits = self._prepare_visits(visits)
        if self.cfg.inductive:
            metrics = self._run_inductive(visits, latent, events)
        else:
            metrics = self._run_transductive(visits, latent, events)
        self._print_evaluation_summary(metrics)
        return metrics

    @staticmethod
    def _print_evaluation_summary(metrics: Dict[str, Any]) -> None:
        """Print a tier-labeled comparison table: GRM vs baselines, with Δ vs best baseline.

        Goal: make it impossible to confuse transductive ("diagnostic") and inductive
        ("deployable") numbers, and surface the apparent gains over the strongest baseline.
        """

        tier = metrics.get("evaluation_tier", "unspecified_tier")
        tier_label = {
            "transductive_diagnostic":
                "TRANSDUCTIVE DIAGNOSTIC  (graph saw all visits during decomposition)",
            "inductive_deployable_prediction":
                "INDUCTIVE DEPLOYABLE  (test subjects disjoint from training)",
        }.get(tier, f"TIER: {tier}")

        reg = metrics.get("regression", {})
        cls = metrics.get("classification", {})
        onset = metrics.get("flare_onset_classification", {})

        reg_order = [
            "grm_ridge", "grm_plus_lag_ridge", "pca_ridge", "pca_plus_lag_ridge",
            "takens_ridge", "takens_plus_lag_ridge",
            "smooth_rbf_kernel_ridge", "raw_random_forest",
            "naive_current_score", "persistence_yesterday_score",
        ]
        cls_order = [
            "grm_logistic", "grm_logistic_calibrated", "grm_plus_lag_logistic",
            "pca_logistic", "pca_plus_lag_logistic",
            "takens_logistic", "takens_plus_lag_logistic",
            "smooth_rbf_kernel_ridge", "raw_random_forest", "naive_current_score", "persistence_yesterday_flare",
        ]
        onset_order = [
            "grm_logistic", "grm_plus_lag_logistic",
            "pca_logistic", "pca_plus_lag_logistic",
            "takens_logistic", "takens_plus_lag_logistic",
            "lag_only_logistic",
            "raw_random_forest", "naive_marginal",
        ]
        reg_metrics = ["r2", "rmse", "mae"]
        cls_metrics = ["roc_auc", "brier", "log_loss"]

        def _fmt(v: Any) -> str:
            if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
                return "    -"
            try:
                return f"{float(v):7.4f}"
            except (TypeError, ValueError):
                return "    -"

        def _row(name: str, d: Dict[str, Any], cols: List[str]) -> str:
            cells = [_fmt(d.get(c)) for c in cols]
            return f"  {name:<32} " + "  ".join(cells)

        def _delta_row(grm_d: Dict[str, Any], best_d: Dict[str, Any], cols: List[str], invert: List[bool]) -> str:
            cells = []
            for c, inv in zip(cols, invert):
                try:
                    g = float(grm_d.get(c)); b = float(best_d.get(c))
                    d = (b - g) if inv else (g - b)
                    cells.append(f"{d:+7.4f}")
                except (TypeError, ValueError):
                    cells.append("    -")
            return f"  {'Δ GRM vs best baseline':<32} " + "  ".join(cells)

        def _best_baseline(table: Dict[str, Any], baseline_keys: List[str], score_key: str, higher_is_better: bool) -> Dict[str, Any]:
            cands = [(k, table[k]) for k in baseline_keys if k in table and table[k]]
            if not cands:
                return {}
            scored = [(k, d, d.get(score_key)) for k, d in cands if d.get(score_key) is not None]
            if not scored:
                return {}
            best = max(scored, key=lambda t: (t[2] if higher_is_better else -t[2]))
            return best[1]

        bar = "=" * 86
        print()
        print(bar)
        print(f"  EVALUATION SUMMARY — {tier_label}")
        print(bar)

        if reg:
            print(f"\n  REGRESSION (target=next_day_score)")
            print(f"  {'predictor':<32} {'R^2':>7}  {'RMSE':>7}  {'MAE':>7}")
            print(f"  {'-' * 32} {'-' * 7}  {'-' * 7}  {'-' * 7}")
            for k in reg_order:
                if k in reg and reg[k]:
                    print(_row(k, reg[k], reg_metrics))
            best_reg = _best_baseline(
                reg, ["pca_ridge", "pca_plus_lag_ridge", "takens_ridge", "takens_plus_lag_ridge", "smooth_rbf_kernel_ridge", "raw_random_forest", "naive_current_score", "persistence_yesterday_score"],
                "r2", higher_is_better=True,
            )
            headline_reg = reg.get("grm_plus_lag_ridge") or reg.get("grm_ridge")
            if headline_reg and best_reg:
                print(_delta_row(headline_reg, best_reg, reg_metrics, invert=[False, True, True]))

        if cls:
            print(f"\n  CLASSIFICATION (target=flare_next_day)")
            print(f"  {'predictor':<32} {'AUC':>7}  {'Brier':>7}  {'LogLs':>7}")
            print(f"  {'-' * 32} {'-' * 7}  {'-' * 7}  {'-' * 7}")
            for k in cls_order:
                if k in cls and cls[k]:
                    print(_row(k, cls[k], cls_metrics))
            best_cls = _best_baseline(
                cls, ["pca_logistic", "pca_plus_lag_logistic", "takens_logistic", "takens_plus_lag_logistic", "smooth_rbf_kernel_ridge", "raw_random_forest", "naive_current_score", "persistence_yesterday_flare"],
                "roc_auc", higher_is_better=True,
            )
            grm_for_delta = cls.get("grm_plus_lag_logistic") or cls.get("grm_logistic_calibrated") or cls.get("grm_logistic")
            if grm_for_delta and best_cls:
                print(_delta_row(grm_for_delta, best_cls, cls_metrics, invert=[False, True, True]))

        if onset:
            n_pos = onset.get("n_test_positive", "?")
            n_elig = onset.get("n_test_eligible", "?")
            marginal = onset.get("train_marginal")
            marginal_str = f"{float(marginal):.3f}" if marginal is not None else "?"
            print(f"\n  CLASSIFICATION (target=flare_onset; today=0 -> tomorrow=1)")
            print(f"  full eligible test rows: {n_elig}, positives: {n_pos}, train marginal: {marginal_str}")
            print(f"  {'predictor':<32} {'AUC':>7}  {'Brier':>7}  {'LogLs':>7}")
            print(f"  {'-' * 32} {'-' * 7}  {'-' * 7}  {'-' * 7}")
            for k in onset_order:
                if k in onset and isinstance(onset[k], dict) and onset[k]:
                    print(_row(k, onset[k], cls_metrics))
            best_onset = _best_baseline(
                onset, ["pca_logistic", "pca_plus_lag_logistic", "takens_logistic", "takens_plus_lag_logistic", "lag_only_logistic", "raw_random_forest", "naive_marginal"],
                "roc_auc", higher_is_better=True,
            )
            grm_for_delta = onset.get("grm_plus_lag_logistic") or onset.get("grm_logistic")
            if grm_for_delta and best_onset:
                print(_delta_row(grm_for_delta, best_onset, cls_metrics, invert=[False, True, True]))

            hard = onset.get("hard_subset_flare_today_0", {}) or {}
            if hard and hard.get("n_test_eligible", 0) > 0:
                n_hard = hard.get("n_test_eligible", "?")
                n_hard_pos = hard.get("n_test_positive", "?")
                print(f"\n  HARD SUBSET (flare_today=0 only; the genuinely-predictive task)")
                print(f"  hard subset rows: {n_hard}, positives: {n_hard_pos}")
                print(f"  {'predictor':<32} {'AUC':>7}  {'Brier':>7}  {'LogLs':>7}")
                print(f"  {'-' * 32} {'-' * 7}  {'-' * 7}  {'-' * 7}")
                for k in onset_order:
                    if k in hard and isinstance(hard[k], dict) and hard[k]:
                        print(_row(k, hard[k], cls_metrics))
                best_hard = _best_baseline(
                    hard, ["pca_logistic", "pca_plus_lag_logistic", "takens_logistic", "takens_plus_lag_logistic", "lag_only_logistic", "raw_random_forest", "naive_marginal"],
                    "roc_auc", higher_is_better=True,
                )
                grm_hard = hard.get("grm_plus_lag_logistic") or hard.get("grm_logistic")
                if grm_hard and best_hard:
                    print(_delta_row(grm_hard, best_hard, cls_metrics, invert=[False, True, True]))

        aliased = metrics.get("aliased_subset_evaluation", {}) or {}
        if aliased and aliased.get("n_eligible", 0) >= 10:
            n_a = aliased.get("n_eligible")
            print(f"\n  ALIASED-PAIR SUBSET (today's obs alias across regimes; futures diverge)")
            print(f"  eligible test rows: {n_a}")
            a_reg = aliased.get("regression", {})
            if a_reg:
                print(f"  REGRESSION (target=next_day_score, aliased subset)")
                print(f"  {'predictor':<32} {'R^2':>7}  {'RMSE':>7}  {'MAE':>7}")
                print(f"  {'-' * 32} {'-' * 7}  {'-' * 7}  {'-' * 7}")
                for k in ["grm_plus_lag_ridge", "pca_plus_lag_ridge", "takens_plus_lag_ridge", "raw_random_forest", "naive_current_score", "persistence_yesterday_score"]:
                    if k in a_reg and a_reg[k]:
                        print(_row(k, a_reg[k], reg_metrics))
                best_a = _best_baseline(
                    a_reg, ["pca_plus_lag_ridge", "takens_plus_lag_ridge", "raw_random_forest", "naive_current_score", "persistence_yesterday_score"],
                    "r2", higher_is_better=True,
                )
                grm_a = a_reg.get("grm_plus_lag_ridge")
                if grm_a and best_a:
                    print(_delta_row(grm_a, best_a, reg_metrics, invert=[False, True, True]))
            a_cls = aliased.get("classification", {})
            if a_cls:
                print(f"  CLASSIFICATION (target=flare_next_day, aliased subset)")
                print(f"  {'predictor':<32} {'AUC':>7}  {'Brier':>7}  {'LogLs':>7}")
                print(f"  {'-' * 32} {'-' * 7}  {'-' * 7}  {'-' * 7}")
                for k in ["grm_plus_lag_logistic", "pca_plus_lag_logistic", "takens_plus_lag_logistic", "raw_random_forest", "naive_current_score", "persistence_yesterday_flare"]:
                    if k in a_cls and a_cls[k]:
                        print(_row(k, a_cls[k], cls_metrics))
                best_ac = _best_baseline(
                    a_cls, ["pca_plus_lag_logistic", "takens_plus_lag_logistic", "raw_random_forest", "naive_current_score", "persistence_yesterday_flare"],
                    "roc_auc", higher_is_better=True,
                )
                grm_ac = a_cls.get("grm_plus_lag_logistic")
                if grm_ac and best_ac:
                    print(_delta_row(grm_ac, best_ac, cls_metrics, invert=[False, True, True]))

        const = metrics.get("constitution_recovery", {})
        if const and const.get("axes"):
            axes = const["axes"]
            print(f"\n  CONSTITUTION RECOVERY (per-subject aggregates -> stable constitution axes)")
            print(f"  train subjects: {const.get('n_train_subjects', '?')}, test subjects: {const.get('n_test_subjects', '?')}")
            head = "  {:<32} ".format("predictor") + "  ".join(f"{a.replace('constitution_',''):>10}" for a in axes) + "    mean"
            print(head)
            print("  " + "-" * 32 + " " + "  ".join("-" * 10 for _ in axes) + "  ------")
            for k, label in [
                ("grm_aggregate_ridge", "visit_grm_aggregate_ridge"),
                ("subject_graph_grm_ridge", "subject_graph_grm_ridge"),
                ("raw_aggregate_ridge", "raw_aggregate_ridge"),
            ]:
                row_metrics = const.get(k, {})
                if not row_metrics:
                    continue
                cells = []
                for a in axes:
                    r2 = row_metrics.get(a, {}).get("r2")
                    cells.append(f"{r2:10.4f}" if r2 is not None else f"{'-':>10}")
                mean_r2 = const.get("mean_r2", {}).get(k)
                mean_s = f"{mean_r2:6.4f}" if mean_r2 is not None else "    -"
                print(f"  {label:<32} " + "  ".join(cells) + f"  {mean_s}")
            grm_mean = const.get("mean_r2", {}).get("grm_aggregate_ridge")
            raw_mean = const.get("mean_r2", {}).get("raw_aggregate_ridge")
            if grm_mean is not None and raw_mean is not None:
                delta = grm_mean - raw_mean
                print(f"  {'Δ GRM vs raw aggregate (mean R²)':<32} " + "  ".join(" " * 10 for _ in axes) + f"  {delta:+6.4f}")

        note = metrics.get("tier_note") or metrics.get("interpretation_guardrail")
        if note:
            print()
            print(f"  NOTE: {note}")
        print(bar)
        print()

    def _run_transductive(self, visits, latent, events) -> Dict:
        self._visit_index = visits[["visit_id", "subject_id", "day"]].copy()
        X_obs, feature_names = self._make_observation_matrix(visits)
        self._X_obs = X_obs
        W = self._build_visit_graph(visits, X_obs, events)
        eigenvalues, eigenvectors = self._spectral_decomposition(W)
        eigenvectors = canonicalize_eigvec_signs(eigenvectors)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        embeddings = self._make_grm_embeddings(eigenvalues, eigenvectors)

        embeddings_df = self._make_embeddings_df(visits, embeddings)
        metrics, predictions_df = self._evaluate(visits, embeddings, latent)
        self._fit_embedding_surrogate(X_obs, embeddings)
        feature_modes_df = self._feature_mode_correlations(visits, embeddings, feature_names)
        self._write_outputs(embeddings_df, feature_modes_df, metrics, predictions_df)
        self._save_model()
        return metrics

    def _run_inductive(self, visits: pd.DataFrame, latent: Optional[pd.DataFrame], events: Optional[pd.DataFrame]) -> Dict:
        """Strict inductive evaluation: split subjects first, fit everything on train-only.

        Pipeline:
          1. Subject-level split (GroupShuffleSplit semantics; seed-controlled).
          2. Fit obs_preprocessor, NN index, visit graph, eigenbasis, embedding surrogate,
             ridge head, logistic head, Procrustes R --- all on TRAIN subjects only.
          3. Project TEST-subject observations via cfg.projection ('surrogate' | 'nystrom').
          4. Score test subjects (regression R², classification AUC, baselines, calibration,
             out-of-sample latent recovery).
          5. Persist the train-only model with manifest extra {inductive:True, ...} and write
             inductive_eval_metrics.json next to the standard CSV outputs.
        """

        if self.cfg.projection not in {"surrogate", "nystrom"}:
            raise ValueError(f"Unknown projection: {self.cfg.projection!r}; must be 'surrogate' or 'nystrom'.")

        # 1. Subject split.
        rng = np.random.default_rng(self.cfg.random_seed)
        all_subjects = np.array(sorted(visits["subject_id"].unique()))
        rng.shuffle(all_subjects)
        n_test = max(1, int(round(self.cfg.test_size * len(all_subjects))))
        if n_test >= len(all_subjects):
            raise ValueError(f"test_size={self.cfg.test_size} leaves no training subjects.")
        test_subjects = set(int(s) for s in all_subjects[:n_test])
        train_subjects = set(int(s) for s in all_subjects[n_test:])

        train_mask = visits["subject_id"].isin(train_subjects).to_numpy()
        train_visits = visits[train_mask].sort_values(["subject_id", "day"]).reset_index(drop=True)
        test_visits = visits[~train_mask].sort_values(["subject_id", "day"]).reset_index(drop=True)
        train_visits["visit_id"] = np.arange(len(train_visits))
        test_visits["visit_id"] = np.arange(len(train_visits), len(train_visits) + len(test_visits))

        print(
            f"[inductive] {len(train_subjects)} train subjects ({len(train_visits)} visits) / "
            f"{len(test_subjects)} test subjects ({len(test_visits)} visits)"
        )

        # 2a. Fit observation preprocessor on train only.
        X_train, feature_names = self._make_observation_matrix(train_visits)
        self._X_obs = X_train

        # 2b. Build train-only graph + eigenbasis.
        W_train = self._build_visit_graph(train_visits, X_train, events)
        eigenvalues, eigenvectors = self._spectral_decomposition(W_train)
        eigenvectors = canonicalize_eigvec_signs(eigenvectors)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        train_embeddings = self._make_grm_embeddings(eigenvalues, eigenvectors)

        # 2c. Fit heads on ALL train embeddings (no within-graph split needed: the held-out
        # set is the disjoint test subjects).
        y_reg_train = train_visits[self.cfg.target_regression].to_numpy(float)
        y_cls_train = train_visits[self.cfg.target_classification].astype(int).to_numpy()
        self.train_idx = np.arange(len(train_visits))
        self.test_idx = None  # No within-graph test split in inductive mode.

        self.ridge_reg = Ridge(alpha=1.0).fit(train_embeddings, y_reg_train)
        if len(np.unique(y_cls_train)) >= 2:
            self.logistic_clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                train_embeddings, y_cls_train
            )
        else:
            self.logistic_clf = None

        # 2d. Surrogate + temperature (both train-only).
        self._fit_embedding_surrogate(X_train, train_embeddings)
        self.flare_temperature = self._fit_flare_temperature(train_embeddings, y_cls_train, self.train_idx)

        # 3. Project test observations.
        X_test_raw = test_visits[self.feature_names].to_numpy(float)
        X_test = self.obs_preprocessor.transform(X_test_raw)

        if self.cfg.projection == "surrogate":
            test_embeddings = surrogate_project(self.embedding_surrogate, X_test)
        else:  # nystrom
            if self.nn_index is None:
                raise RuntimeError(
                    f"Nyström projection requires a KNN-based graph_mode; got {self.cfg.graph_mode!r}."
                )
            test_embeddings = nystrom_extend_arrays(
                X_test,
                nn_index=self.nn_index,
                knn_sigma=float(self.knn_sigma),
                eigenvalues=self.eigenvalues,
                eigenvectors=self.eigenvectors,
                rho=float(self.cfg.rho),
                normalized=bool(self.cfg.use_normalized_laplacian),
                train_degrees=self.train_degrees,
                n_neighbors=int(self.cfg.n_neighbors_inductive),
            )

        # 4. Score test subjects + baselines.
        y_reg_test = test_visits[self.cfg.target_regression].to_numpy(float)
        y_cls_test = test_visits[self.cfg.target_classification].astype(int).to_numpy()

        pred_grm_reg = self.ridge_reg.predict(test_embeddings)
        if self.logistic_clf is not None:
            prob_grm_cls = self.logistic_clf.predict_proba(test_embeddings)[:, 1]
            pred_grm_cls = (prob_grm_cls >= 0.5).astype(int)
            if self.flare_temperature is not None and np.isfinite(self.flare_temperature):
                z_test = self.logistic_clf.decision_function(test_embeddings)
                prob_grm_cls_cal = _sigmoid(z_test / float(self.flare_temperature))
                pred_grm_cls_cal = (prob_grm_cls_cal >= 0.5).astype(int)
            else:
                prob_grm_cls_cal = None
                pred_grm_cls_cal = None
        else:
            prob_grm_cls = np.full(len(test_visits), 0.5)
            pred_grm_cls = np.zeros(len(test_visits), dtype=int)
            prob_grm_cls_cal = None
            pred_grm_cls_cal = None

        # Raw-observation baseline: fit RF on train_raw, predict on test_raw. Both rows
        # come through the SAME preprocessor (fit on train) to keep the inductive contract.
        # Uses the same feature set the GRM head sees so the comparison is fair.
        raw_cols = self.feature_names + ["global_dysregulation_score"]
        raw_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        train_raw = raw_pipe.fit_transform(train_visits[raw_cols].to_numpy(float))
        test_raw = raw_pipe.transform(test_visits[raw_cols].to_numpy(float))
        raw_rf_reg = RandomForestRegressor(n_estimators=100, min_samples_leaf=4, random_state=42, n_jobs=-1).fit(
            train_raw, y_reg_train
        )
        pred_raw_reg = raw_rf_reg.predict(test_raw)
        if len(np.unique(y_cls_train)) >= 2:
            raw_rf_cls = RandomForestClassifier(
                n_estimators=100, min_samples_leaf=4, random_state=42, n_jobs=-1, class_weight="balanced"
            ).fit(train_raw, y_cls_train)
            prob_raw_cls = raw_rf_cls.predict_proba(test_raw)[:, 1]
            pred_raw_cls = (prob_raw_cls >= 0.5).astype(int)
        else:
            prob_raw_cls = np.full(len(test_visits), 0.5)
            pred_raw_cls = np.zeros(len(test_visits), dtype=int)

        pred_smooth_reg, prob_smooth_cls = self._fit_smooth_rbf_baseline(
            train_raw, y_reg_train, y_cls_train, test_raw
        )

        # Naive baseline: just use the current dysregulation score.
        pred_naive_reg = test_visits["global_dysregulation_score"].to_numpy(float)
        pred_naive_reg = np.nan_to_num(pred_naive_reg, nan=float(np.nanmedian(y_reg_train)))
        naive_threshold = float(np.nanmedian(y_reg_train))
        pred_naive_cls = (pred_naive_reg >= naive_threshold).astype(int)
        prob_naive_cls = np.clip(
            pred_naive_reg / max(float(np.nanmax(y_reg_train)), 1e-9), 0, 1
        )

        persistence = self._persistence_baseline(
            test_visits, train_visits,
            self.cfg.target_classification, self.cfg.target_regression,
        )

        # GRM + lag head — fair deployment-style competitor against persistence.
        fill_score = float(np.nanmedian(y_reg_train))
        fill_flare = float(np.nanmean(y_cls_train))
        X_train_grm_lag = self._augment_with_lag(train_embeddings, train_visits, fill_score, fill_flare)
        X_test_grm_lag = self._augment_with_lag(test_embeddings, test_visits, fill_score, fill_flare)
        ridge_grm_lag = Ridge(alpha=1.0).fit(X_train_grm_lag, y_reg_train)
        pred_grm_lag_reg = ridge_grm_lag.predict(X_test_grm_lag)
        if len(np.unique(y_cls_train)) >= 2:
            log_grm_lag = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_grm_lag, y_cls_train)
            prob_grm_lag_cls = log_grm_lag.predict_proba(X_test_grm_lag)[:, 1]
            pred_grm_lag_cls = (prob_grm_lag_cls >= 0.5).astype(int)
        else:
            prob_grm_lag_cls = np.full(len(test_visits), 0.5)
            pred_grm_lag_cls = np.zeros(len(test_visits), dtype=int)

        # PCA baseline: linear projection into same dimensionality as GRM modes.
        # Fit on train_raw only; transform both train and test.
        pca = PCA(n_components=self.cfg.n_modes, random_state=self.cfg.random_seed)
        pca_train = pca.fit_transform(train_raw)
        pca_test = pca.transform(test_raw)
        pca_ridge = Ridge(alpha=1.0).fit(pca_train, y_reg_train)
        pred_pca_reg = pca_ridge.predict(pca_test)
        if len(np.unique(y_cls_train)) >= 2:
            pca_log = LogisticRegression(max_iter=2000, class_weight="balanced").fit(pca_train, y_cls_train)
            prob_pca_cls = pca_log.predict_proba(pca_test)[:, 1]
            pred_pca_cls = (prob_pca_cls >= 0.5).astype(int)
        else:
            prob_pca_cls = np.full(len(test_visits), 0.5)
            pred_pca_cls = np.zeros(len(test_visits), dtype=int)
        X_train_pca_lag = self._augment_with_lag(pca_train, train_visits, fill_score, fill_flare)
        X_test_pca_lag = self._augment_with_lag(pca_test, test_visits, fill_score, fill_flare)
        pca_lag_ridge = Ridge(alpha=1.0).fit(X_train_pca_lag, y_reg_train)
        pred_pca_lag_reg = pca_lag_ridge.predict(X_test_pca_lag)
        if len(np.unique(y_cls_train)) >= 2:
            pca_lag_log = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_pca_lag, y_cls_train)
            prob_pca_lag_cls = pca_lag_log.predict_proba(X_test_pca_lag)[:, 1]
            pred_pca_lag_cls = (prob_pca_lag_cls >= 0.5).astype(int)
        else:
            prob_pca_lag_cls = np.full(len(test_visits), 0.5)
            pred_pca_lag_cls = np.zeros(len(test_visits), dtype=int)

        # Delay-embedded (Takens) baselines.
        takens_train = self._build_delay_embedding(train_raw, train_visits, self.cfg.delay_embedding_k)
        takens_test = self._build_delay_embedding(test_raw, test_visits, self.cfg.delay_embedding_k)
        takens_ridge_m = Ridge(alpha=1.0).fit(takens_train, y_reg_train)
        pred_takens_reg = takens_ridge_m.predict(takens_test)
        if len(np.unique(y_cls_train)) >= 2:
            takens_log_m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(takens_train, y_cls_train)
            prob_takens_cls = takens_log_m.predict_proba(takens_test)[:, 1]
            pred_takens_cls = (prob_takens_cls >= 0.5).astype(int)
        else:
            prob_takens_cls = np.full(len(test_visits), 0.5)
            pred_takens_cls = np.zeros(len(test_visits), dtype=int)
        X_train_takens_lag = self._augment_with_lag(takens_train, train_visits, fill_score, fill_flare)
        X_test_takens_lag = self._augment_with_lag(takens_test, test_visits, fill_score, fill_flare)
        takens_lag_ridge_m = Ridge(alpha=1.0).fit(X_train_takens_lag, y_reg_train)
        pred_takens_lag_reg = takens_lag_ridge_m.predict(X_test_takens_lag)
        if len(np.unique(y_cls_train)) >= 2:
            takens_lag_log_m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_takens_lag, y_cls_train)
            prob_takens_lag_cls = takens_lag_log_m.predict_proba(X_test_takens_lag)[:, 1]
            pred_takens_lag_cls = (prob_takens_lag_cls >= 0.5).astype(int)
        else:
            prob_takens_lag_cls = np.full(len(test_visits), 0.5)
            pred_takens_lag_cls = np.zeros(len(test_visits), dtype=int)

        # Flare-onset secondary target. Eligible rows = where flare_today is known.
        y_onset_train_raw = train_visits[self.cfg.target_classification_onset].astype(float).to_numpy()
        y_onset_test_raw = test_visits[self.cfg.target_classification_onset].astype(float).to_numpy()
        tr_onset_valid = ~np.isnan(y_onset_train_raw)
        te_onset_valid = ~np.isnan(y_onset_test_raw)
        y_onset_train = np.where(tr_onset_valid, np.nan_to_num(y_onset_train_raw, nan=0.0), 0).astype(int)
        y_onset_test = np.where(te_onset_valid, np.nan_to_num(y_onset_test_raw, nan=0.0), 0).astype(int)
        flare_today_test_arr = test_visits["flare_persistence_today"].astype(float).to_numpy()[te_onset_valid]
        onset_block = self._fit_and_score_onset(
            y_onset_train[tr_onset_valid], y_onset_test[te_onset_valid],
            train_embeddings[tr_onset_valid], test_embeddings[te_onset_valid],
            X_train_grm_lag[tr_onset_valid], X_test_grm_lag[te_onset_valid],
            train_raw[tr_onset_valid], test_raw[te_onset_valid],
            flare_today_test_arr,
            X_pca_train=pca_train[tr_onset_valid], X_pca_test=pca_test[te_onset_valid],
            X_pca_lag_train=X_train_pca_lag[tr_onset_valid], X_pca_lag_test=X_test_pca_lag[te_onset_valid],
            X_takens_train=takens_train[tr_onset_valid], X_takens_test=takens_test[te_onset_valid],
            X_takens_lag_train=X_train_takens_lag[tr_onset_valid], X_takens_lag_test=X_test_takens_lag[te_onset_valid],
        )

        # 5. Out-of-sample latent recovery: fit Procrustes on train, apply to test.
        latent_recovery = self._latent_recovery_inductive(
            train_visits, train_embeddings, test_visits, test_embeddings, latent
        )

        metrics: Dict[str, Any] = {
            "manifest": "model/manifest.json",
            "evaluation_mode": "inductive",
            "evaluation_tier": "inductive_deployable_prediction",
            "projection": self.cfg.projection,
            "n_train_subjects": int(len(train_subjects)),
            "n_test_subjects": int(len(test_subjects)),
            "n_train_visits": int(len(train_visits)),
            "n_test_visits": int(len(test_visits)),
            "regression": {
                "grm_ridge": self._reg_metrics(y_reg_test, pred_grm_reg),
                "grm_plus_lag_ridge": self._reg_metrics(y_reg_test, pred_grm_lag_reg),
                "pca_ridge": self._reg_metrics(y_reg_test, pred_pca_reg),
                "pca_plus_lag_ridge": self._reg_metrics(y_reg_test, pred_pca_lag_reg),
                "takens_ridge": self._reg_metrics(y_reg_test, pred_takens_reg),
                "takens_plus_lag_ridge": self._reg_metrics(y_reg_test, pred_takens_lag_reg),
                "smooth_rbf_kernel_ridge": self._reg_metrics(y_reg_test, pred_smooth_reg),
                "raw_random_forest": self._reg_metrics(y_reg_test, pred_raw_reg),
                "naive_current_score": self._reg_metrics(y_reg_test, pred_naive_reg),
                "persistence_yesterday_score": self._reg_metrics(y_reg_test, persistence["score_pred"]),
            },
            "classification": {
                "grm_logistic": self._cls_metrics(y_cls_test, pred_grm_cls, prob_grm_cls),
                "grm_logistic_calibrated": (
                    self._cls_metrics(y_cls_test, pred_grm_cls_cal, prob_grm_cls_cal)
                    if prob_grm_cls_cal is not None else {}
                ),
                "grm_plus_lag_logistic": self._cls_metrics(y_cls_test, pred_grm_lag_cls, prob_grm_lag_cls),
                "pca_logistic": self._cls_metrics(y_cls_test, pred_pca_cls, prob_pca_cls),
                "pca_plus_lag_logistic": self._cls_metrics(y_cls_test, pred_pca_lag_cls, prob_pca_lag_cls),
                "takens_logistic": self._cls_metrics(y_cls_test, pred_takens_cls, prob_takens_cls),
                "takens_plus_lag_logistic": self._cls_metrics(y_cls_test, pred_takens_lag_cls, prob_takens_lag_cls),
                "smooth_rbf_kernel_ridge": self._cls_metrics(
                    y_cls_test, (prob_smooth_cls >= 0.5).astype(int), prob_smooth_cls
                ),
                "raw_random_forest": self._cls_metrics(y_cls_test, pred_raw_cls, prob_raw_cls),
                "naive_current_score": self._cls_metrics(y_cls_test, pred_naive_cls, prob_naive_cls),
                "persistence_yesterday_flare": self._cls_metrics(
                    y_cls_test, persistence["flare_pred"], persistence["flare_prob"]
                ),
            },
            "flare_onset_classification": onset_block,
            "aliased_subset_evaluation": self._evaluate_aliased_subset(
                test_visits, y_reg_test, y_cls_test,
                pred_grm_lag_reg, prob_grm_lag_cls,
                pred_raw_reg, prob_raw_cls,
                pred_naive_reg, prob_naive_cls,
                persistence,
                pred_pca_lag_reg=pred_pca_lag_reg,
                prob_pca_lag_cls=prob_pca_lag_cls,
                pred_takens_lag_reg=pred_takens_lag_reg,
                prob_takens_lag_cls=prob_takens_lag_cls,
            ),
            "constitution_recovery": self._evaluate_constitution_recovery(
                train_visits, train_embeddings, test_visits, test_embeddings,
            ),
            "spectral_signal_concentration": self._spectral_signal_concentration(test_visits, test_embeddings),
            "parsimony": self._parsimony_summary(),
            "flare_temperature": float(self.flare_temperature) if self.flare_temperature is not None else None,
            "latent_recovery": latent_recovery,
            "interpretation_guardrail": (
                "Strict inductive evaluation: test subjects are disjoint from training and were "
                "never seen by the graph, eigenbasis, or any fitted head. Compare against the "
                "transductive metrics to see how much of the apparent signal is graph-leak."
            ),
        }

        # 6. Build combined embeddings frame + test-only predictions frame.
        embeddings_df = pd.concat(
            [
                self._make_embeddings_df(train_visits, train_embeddings).assign(split="train"),
                self._make_embeddings_df(test_visits, test_embeddings).assign(split="test"),
            ],
            ignore_index=True,
        )

        pred_df = test_visits[
            ["visit_id", "subject_id", "day", self.cfg.target_regression, self.cfg.target_classification]
        ].copy()
        pred_df["split"] = "test"
        pred_df["pred_grm_next_score"] = pred_grm_reg
        pred_df["pred_grm_flare_prob"] = prob_grm_cls
        if prob_grm_cls_cal is not None:
            pred_df["pred_grm_flare_prob_calibrated"] = prob_grm_cls_cal
        pred_df["pred_raw_next_score"] = pred_raw_reg
        pred_df["pred_raw_flare_prob"] = prob_raw_cls
        pred_df["pred_naive_next_score"] = pred_naive_reg
        pred_df["pred_naive_flare_prob"] = prob_naive_cls
        pred_df["pred_persistence_next_score"] = persistence["score_pred"]
        pred_df["pred_persistence_flare_prob"] = persistence["flare_prob"]
        pred_df["pred_grm_plus_lag_next_score"] = pred_grm_lag_reg
        pred_df["pred_grm_plus_lag_flare_prob"] = prob_grm_lag_cls

        feature_modes_df = self._feature_mode_correlations(train_visits, train_embeddings, feature_names)

        # 7. Write outputs. The persisted model is the train-only fit; _visit_index for
        # save_model is restricted to train rows so it aligns with eigenvectors.
        self._visit_index = train_visits[["visit_id", "subject_id", "day"]].copy()
        self._write_outputs(embeddings_df, feature_modes_df, metrics, pred_df)
        with open(self.output_dir / "inductive_eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        manifest_extra = {
            "inductive": True,
            "projection": self.cfg.projection,
            "n_train_subjects": int(len(train_subjects)),
            "n_test_subjects": int(len(test_subjects)),
            "test_subjects": sorted(test_subjects),
        }
        self._save_model(manifest_extra=manifest_extra)
        self._write_inductive_plots(metrics, pred_df)
        return metrics

    def _latent_recovery_inductive(
        self,
        train_visits: pd.DataFrame,
        train_embeddings: np.ndarray,
        test_visits: pd.DataFrame,
        test_embeddings: np.ndarray,
        latent: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Fit Procrustes on train, evaluate alignment in-sample (train) AND out-of-sample (test)."""

        if latent is None:
            return {}

        def _merge_latent(v: pd.DataFrame) -> np.ndarray:
            merged = v[["visit_id", "subject_id", "day"]].merge(
                latent[["subject_id", "day"] + LATENT_NAMES], on=["subject_id", "day"], how="left"
            )
            return merged[LATENT_NAMES].to_numpy(float)

        Z_train_raw = _merge_latent(train_visits)
        Z_test_raw = _merge_latent(test_visits)
        z_imputer = SimpleImputer(strategy="median").fit(Z_train_raw)
        z_scaler = StandardScaler().fit(z_imputer.transform(Z_train_raw))
        Z_train = z_scaler.transform(z_imputer.transform(Z_train_raw))
        Z_test = z_scaler.transform(z_imputer.transform(Z_test_raw))

        e_scaler = StandardScaler().fit(train_embeddings)
        E_train = e_scaler.transform(train_embeddings)
        E_test = e_scaler.transform(test_embeddings)

        q = min(E_train.shape[1], Z_train.shape[1])
        R, _ = orthogonal_procrustes(E_train[:, :q], Z_train[:, :q])
        self.procrustes_R = R

        def _corrs(E_aligned: np.ndarray, Z: np.ndarray) -> List[float]:
            return [
                float(np.corrcoef(E_aligned[:, j], Z[:, j])[0, 1]) if E_aligned.shape[0] > 1 else float("nan")
                for j in range(q)
            ]

        train_corrs = _corrs(E_train[:, :q] @ R, Z_train[:, :q])
        test_corrs = _corrs(E_test[:, :q] @ R, Z_test[:, :q])
        return {
            "in_sample_train": {
                "mean_abs_aligned_correlation": float(np.nanmean(np.abs(train_corrs))),
                "aligned_correlations": train_corrs,
            },
            "out_of_sample_test": {
                "mean_abs_aligned_correlation": float(np.nanmean(np.abs(test_corrs))),
                "aligned_correlations": test_corrs,
            },
            "note": (
                "Synthetic-only metric. Procrustes rotation fit on train embeddings vs train latents, "
                "applied to held-out test embeddings vs test latents."
            ),
        }

    def _load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        visits_path = self.input_dir / "visits.csv"
        if not visits_path.exists():
            raise FileNotFoundError(f"Missing {visits_path}. Run grm_tcm_synthetic_generator.py first.")
        visits = pd.read_csv(visits_path)
        latent = pd.read_csv(self.input_dir / "latent_states.csv") if (self.input_dir / "latent_states.csv").exists() else None
        events = pd.read_csv(self.input_dir / "events.csv") if (self.input_dir / "events.csv").exists() else None
        # Subjects.csv is optional — only the constitution-recovery evaluation needs it.
        subjects_path = self.input_dir / "subjects.csv"
        self.subjects: Optional[pd.DataFrame] = pd.read_csv(subjects_path) if subjects_path.exists() else None
        return visits, latent, events

    def _prepare_visits(self, visits: pd.DataFrame) -> pd.DataFrame:
        df = visits.sort_values(["subject_id", "day"]).reset_index(drop=True).copy()
        for col in [self.cfg.target_regression, self.cfg.target_classification]:
            if col not in df.columns:
                raise ValueError(f"Missing target column: {col}")
        df = df.dropna(subset=[self.cfg.target_regression, self.cfg.target_classification]).reset_index(drop=True)
        df["visit_id"] = np.arange(len(df))
        # Persistence-baseline column: yesterday's flare label (= today's flare).
        # Constructed AFTER the NaN drop so consecutive remaining rows define "yesterday".
        # NaN for the first visit per subject — callers fall back to train marginal.
        df["flare_persistence_today"] = df.groupby("subject_id")[self.cfg.target_classification].shift(1)
        df["score_persistence_today"] = df.groupby("subject_id")[self.cfg.target_regression].shift(1)

        # Flare ONSET = (today=0, tomorrow=1). Derived from the primary flare target
        # and its per-subject lag. NA for the first visit per subject (no "today" known).
        flare_today = df["flare_persistence_today"]
        flare_tomorrow = df[self.cfg.target_classification].astype(float)
        onset = ((flare_tomorrow == 1.0) & (flare_today == 0.0)).astype(int)
        df[self.cfg.target_classification_onset] = pd.array(
            np.where(flare_today.isna(), pd.NA, onset.to_numpy()),
            dtype="Int64",
        )
        return df

    def _active_feature_names(self, visits: pd.DataFrame) -> List[str]:
        """Return the feature columns the trainer will actually use.

        Always the 12 continuous OBSERVATION_NAMES. Additionally includes the v2
        qualitative ordinal channels when `include_qualitative_features` is on
        AND those columns are present in visits.csv (so v1 data still works).
        """

        feats = list(OBSERVATION_NAMES)
        if self.cfg.include_qualitative_features:
            for q in QUALITATIVE_FEATURE_NAMES:
                if q in visits.columns:
                    feats.append(q)
        return feats

    def _make_observation_matrix(self, visits: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        feature_names = self._active_feature_names(visits)
        missing = [c for c in feature_names if c not in visits.columns]
        if missing:
            raise ValueError(f"Missing observation columns: {missing}")
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        X = pipe.fit_transform(visits[feature_names].to_numpy(dtype=float))
        self.obs_preprocessor = pipe
        self.feature_names = feature_names
        return X, feature_names

    def _build_visit_graph(self, visits: pd.DataFrame, X: np.ndarray, events: Optional[pd.DataFrame]) -> sparse.csr_matrix:
        valid_modes = {
            "feature_only",
            "feature_only_diffusion",
            "temporal_only",
            "feature_temporal",
            "feature_temporal_treatment",
            "feature_temporal_treatment_subject",
            "random_graph",
        }
        if self.cfg.graph_mode not in valid_modes:
            raise ValueError(f"Unknown graph_mode: {self.cfg.graph_mode}. Choose one of {sorted(valid_modes)}")
        if self.cfg.graph_mode == "random_graph":
            return self._build_random_graph(len(visits))

        n = len(visits)
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        indices: Optional[np.ndarray] = None

        # KNN graph on observation similarity.
        feature_graph_modes = {
            "feature_only",
            "feature_only_diffusion",
            "feature_temporal",
            "feature_temporal_treatment",
            "feature_temporal_treatment_subject",
        }
        if self.cfg.graph_mode in feature_graph_modes:
            nn = NearestNeighbors(n_neighbors=min(self.cfg.n_neighbors + 1, n), metric="euclidean")
            nn.fit(X)
            distances, indices = nn.kneighbors(X)
            nonzero = distances[:, 1:].ravel()
            nonzero = nonzero[nonzero > 0]
            sigma = float(self.cfg.similarity_sigma or (np.median(nonzero) if len(nonzero) else 1.0))
            sigma = max(sigma, 1e-9)
            self.nn_index = nn
            self.knn_sigma = sigma

            for i in range(n):
                for dist, j in zip(distances[i, 1:], indices[i, 1:]):
                    weight = float(np.exp(-(dist ** 2) / (2.0 * sigma ** 2)))
                    rows += [i, int(j)]
                    cols += [int(j), i]
                    vals += [weight, weight]

        # Temporal same-subject edges.
        visit_index = {(int(r.subject_id), int(r.day)): int(r.visit_id) for r in visits.itertuples(index=False)}
        temporal_graph_modes = {
            "temporal_only",
            "feature_temporal",
            "feature_temporal_treatment",
            "feature_temporal_treatment_subject",
        }
        if self.cfg.graph_mode in temporal_graph_modes:
            for r in visits.itertuples(index=False):
                i = int(r.visit_id)
                nxt = (int(r.subject_id), int(r.day) + 1)
                if nxt in visit_index:
                    j = visit_index[nxt]
                    rows += [i, j]
                    cols += [j, i]
                    vals += [self.cfg.temporal_edge_weight, self.cfg.temporal_edge_weight]

            # Weak same-subject smoothness edges up to 3 days away.
            for _, group in visits.groupby("subject_id"):
                ids = group["visit_id"].to_numpy(int)
                days = group["day"].to_numpy(int)
                for a in range(len(ids)):
                    for b in range(a + 2, min(a + 4, len(ids))):
                        dt = abs(int(days[b]) - int(days[a]))
                        weight = float(self.cfg.same_subject_edge_weight * np.exp(-dt / 3.0))
                        rows += [ids[a], ids[b]]
                        cols += [ids[b], ids[a]]
                        vals += [weight, weight]

        # Optional treatment-similarity edges among feature-near treatment visits.
        if (
            self.cfg.graph_mode in {"feature_temporal_treatment", "feature_temporal_treatment_subject"}
            and indices is not None
            and events is not None
            and not events.empty
            and "event_type" in events.columns
        ):
            treatment_keys = {
                (int(r.subject_id), int(r.day))
                for r in events[events["event_type"] == "treatment_event"][["subject_id", "day"]].itertuples(index=False)
            }
            treatment_ids = {visit_index[k] for k in treatment_keys if k in visit_index}
            if len(treatment_ids) > 1:
                for i in treatment_ids:
                    for j in indices[i, 1:]:
                        j = int(j)
                        if j in treatment_ids:
                            rows += [i, j]
                            cols += [j, i]
                            vals += [self.cfg.treatment_edge_weight, self.cfg.treatment_edge_weight]

        # Cross-subject constitution edges. These are deliberately separate from
        # visit KNN edges: subject means define who is constitutionally similar,
        # then same-day visits between similar subjects are weakly tied. This gives
        # the spectral geometry a route to represent durable subject structure
        # without leaking subject IDs as features.
        if self.cfg.graph_mode == "feature_temporal_treatment_subject":
            subj_rows, subj_cols, subj_vals = self._subject_similarity_edges(visits, X)
            rows.extend(subj_rows)
            cols.extend(subj_cols)
            vals.extend(subj_vals)

        W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        W.setdiag(0.0)
        W.eliminate_zeros()
        W = W.maximum(W.T)

        if self.cfg.graph_mode == "feature_only_diffusion":
            # Density-corrected diffusion-map normalization. This graph mode is
            # meant for geometry recovery: alpha=1 reduces sampling-density bias
            # so the graph Laplacian better approximates the Laplace-Beltrami
            # operator instead of the density-weighted diffusion operator.
            alpha = float(self.cfg.diffusion_alpha)
            if alpha != 0.0:
                q = np.maximum(np.asarray(W.sum(axis=1)).ravel(), 1e-12)
                Q = sparse.diags(q ** (-alpha))
                W = Q @ W @ Q
                W.setdiag(0.0)
                W.eliminate_zeros()
                W = W.maximum(W.T)

        return W

    def _subject_similarity_edges(self, visits: pd.DataFrame, X: np.ndarray) -> Tuple[List[int], List[int], List[float]]:
        """Build weak same-day edges between constitutionally similar subjects.

        Subject similarity is estimated from each subject's mean standardized
        observation vector. Edges then connect visits at the same day between
        neighboring subjects, preserving longitudinal phase while making stable
        subject-level similarity visible to the visit graph.
        """

        subjects = np.array(sorted(int(s) for s in visits["subject_id"].unique()))
        if len(subjects) < 2:
            return [], [], []

        subj_to_pos = {sid: pos for pos, sid in enumerate(subjects)}
        subj_means = np.zeros((len(subjects), X.shape[1]), dtype=float)
        for sid, group in visits.groupby("subject_id"):
            subj_means[subj_to_pos[int(sid)]] = X[group.index.to_numpy()].mean(axis=0)

        n_neighbors = min(max(int(self.cfg.subject_similarity_neighbors), 1) + 1, len(subjects))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(subj_means)
        distances, indices = nn.kneighbors(subj_means)
        nonzero = distances[:, 1:].ravel()
        nonzero = nonzero[nonzero > 0]
        sigma = float(np.median(nonzero) if len(nonzero) else 1.0)
        sigma = max(sigma, 1e-9)

        by_key = {
            (int(row.subject_id), int(row.day)): int(row.visit_id)
            for row in visits[["visit_id", "subject_id", "day"]].itertuples(index=False)
        }
        days_by_subject = {
            int(sid): set(int(day) for day in group["day"].to_numpy(int))
            for sid, group in visits.groupby("subject_id")
        }

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        scale = float(self.cfg.subject_similarity_edge_weight)
        if scale <= 0.0:
            return rows, cols, vals

        for a_pos, sid_a in enumerate(subjects):
            days_a = days_by_subject[int(sid_a)]
            for dist, b_pos in zip(distances[a_pos, 1:], indices[a_pos, 1:]):
                sid_b = int(subjects[int(b_pos)])
                shared_days = days_a & days_by_subject[sid_b]
                if not shared_days:
                    continue
                weight = scale * float(np.exp(-(float(dist) ** 2) / (2.0 * sigma ** 2)))
                for day in shared_days:
                    i = by_key[(int(sid_a), int(day))]
                    j = by_key[(sid_b, int(day))]
                    rows += [i, j]
                    cols += [j, i]
                    vals += [weight, weight]
        return rows, cols, vals

    def _build_random_graph(self, n: int) -> sparse.csr_matrix:
        rng = np.random.default_rng(self.cfg.random_seed)
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        degree = min(max(self.cfg.n_neighbors, 2), max(n - 1, 1))
        for i in range(n):
            choices = rng.choice(np.delete(np.arange(n), i), size=degree, replace=False)
            for j in choices:
                rows += [i, int(j)]
                cols += [int(j), i]
                vals += [1.0, 1.0]
        W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        W.setdiag(0.0)
        W.eliminate_zeros()
        return W.maximum(W.T)

    def _spectral_decomposition(self, W: sparse.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
        n = W.shape[0]
        degrees = np.asarray(W.sum(axis=1)).ravel()
        degrees = np.maximum(degrees, 1e-12)
        self.train_degrees = degrees.copy()
        if self.cfg.use_normalized_laplacian:
            D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
            L = sparse.eye(n, format="csr") - D_inv_sqrt @ W @ D_inv_sqrt
        else:
            L = sparse.diags(degrees) - W

        # Compute extra eigenvalues beyond n_modes so the diagnostic spectrum
        # plot can show tail behavior past the retained cutoff. The extra
        # eigenvalues are persisted but not used to construct embeddings.
        extra_modes = 24
        k = min(self.cfg.n_modes + extra_modes + 1, n - 2)
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
        order = np.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        self.eigenvalues_full = eigenvalues[1:].copy()
        return eigenvalues[1:self.cfg.n_modes + 1], eigenvectors[:, 1:self.cfg.n_modes + 1]

    def _make_grm_embeddings(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
        # Diffusion-map convention: emb[i, m] = sqrt(g(lambda_m)) * psi_m(i), where
        # g(lambda) = 1 / (1 + rho^2 * lambda) is the GRM filter. Inner products of
        # embeddings then literally reconstruct the GRM kernel:
        #     <emb_i, emb_j> = sum_m g(lambda_m) psi_m(i) psi_m(j) = G_ij
        # which is the propagator from grm_tcm_dynamic_grm.spectral_grm. The previous
        # weight-without-sqrt convention double-counted g under embedding inner products.
        weights = 1.0 / np.sqrt(1.0 + (self.cfg.rho ** 2) * eigenvalues)
        return eigenvectors * weights.reshape(1, -1)

    def _make_embeddings_df(self, visits: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
        df = visits[["visit_id", "subject_id", "day"]].copy()
        for i in range(embeddings.shape[1]):
            df[f"grm_mode_{i + 1}"] = embeddings[:, i]
        return df

    def _evaluate(self, visits: pd.DataFrame, embeddings: np.ndarray, latent: Optional[pd.DataFrame]) -> Tuple[Dict, pd.DataFrame]:
        y_reg = visits[self.cfg.target_regression].to_numpy(float)
        y_cls = visits[self.cfg.target_classification].astype(int).to_numpy()
        groups = visits["subject_id"].to_numpy()
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=self.cfg.test_size, random_state=self.cfg.random_seed).split(embeddings, y_reg, groups))
        self.train_idx = train_idx
        self.test_idx = test_idx

        X_grm = embeddings
        X_raw = self._raw_baseline_matrix(visits)
        pred_grm_reg, self.ridge_reg = self._fit_reg(X_grm, y_reg, train_idx, test_idx, "ridge")
        pred_raw_reg, _ = self._fit_reg(X_raw, y_reg, train_idx, test_idx, "random_forest")
        pred_smooth_reg, prob_smooth_cls = self._fit_smooth_rbf_baseline(
            X_raw[train_idx], y_reg[train_idx], y_cls[train_idx], X_raw[test_idx]
        )
        pred_naive_reg = visits.iloc[test_idx]["global_dysregulation_score"].to_numpy(float)
        pred_naive_reg = np.nan_to_num(pred_naive_reg, nan=float(np.nanmedian(y_reg[train_idx])))

        pred_grm_cls, prob_grm_cls, self.logistic_clf = self._fit_cls(X_grm, y_cls, train_idx, test_idx, "logistic")
        pred_raw_cls, prob_raw_cls, _ = self._fit_cls(X_raw, y_cls, train_idx, test_idx, "random_forest")
        naive_threshold = float(np.nanmedian(y_reg[train_idx]))
        pred_naive_cls = (pred_naive_reg >= naive_threshold).astype(int)
        prob_naive_cls = np.clip(pred_naive_reg / max(float(np.nanmax(y_reg[train_idx])), 1e-9), 0, 1)

        persistence = self._persistence_baseline(
            visits.iloc[test_idx], visits.iloc[train_idx],
            self.cfg.target_classification, self.cfg.target_regression,
        )

        # PCA baseline: linear projection of raw observations into the same
        # dimensionality as GRM modes. Fit on train only to respect the split.
        # This is the Pang-EDR analogue: if PCA matches GRM, the load-bearing
        # signal is linear variance, not graph topology.
        pca = PCA(n_components=self.cfg.n_modes, random_state=self.cfg.random_seed)
        pca.fit(X_raw[train_idx])
        X_pca = pca.transform(X_raw)
        pred_pca_reg, _ = self._fit_reg(X_pca, y_reg, train_idx, test_idx, "ridge")
        pred_pca_cls, prob_pca_cls, _ = self._fit_cls(X_pca, y_cls, train_idx, test_idx, "logistic")

        # GRM + lag head: same GRM coordinates with per-subject target lags appended.
        # This is the "fair deployment" comparison against the persistence baseline.
        fill_score = float(np.nanmedian(y_reg[train_idx]))
        fill_flare = float(np.nanmean(y_cls[train_idx]))
        X_grm_lag = self._augment_with_lag(X_grm, visits, fill_score, fill_flare)
        pred_grm_lag_reg, _ = self._fit_reg(X_grm_lag, y_reg, train_idx, test_idx, "ridge")
        pred_grm_lag_cls, prob_grm_lag_cls, _ = self._fit_cls(X_grm_lag, y_cls, train_idx, test_idx, "logistic")

        # PCA + lag head: same lag augmentation applied to PCA embeddings.
        X_pca_lag = self._augment_with_lag(X_pca, visits, fill_score, fill_flare)
        pred_pca_lag_reg, _ = self._fit_reg(X_pca_lag, y_reg, train_idx, test_idx, "ridge")
        pred_pca_lag_cls, prob_pca_lag_cls, _ = self._fit_cls(X_pca_lag, y_cls, train_idx, test_idx, "logistic")

        # Delay-embedded (Takens) baselines: concatenate last k visits into a fat
        # feature vector.  Tests whether trajectory information improves prediction
        # without building a full graph OOSE.  If takens ≈ static, trajectory signal
        # is too noisy to exploit; if takens >> static, the OOSE refactor is justified.
        X_takens = self._build_delay_embedding(X_raw, visits, self.cfg.delay_embedding_k)
        pred_takens_reg, _ = self._fit_reg(X_takens, y_reg, train_idx, test_idx, "ridge")
        pred_takens_cls, prob_takens_cls, _ = self._fit_cls(X_takens, y_cls, train_idx, test_idx, "logistic")
        X_takens_lag = self._augment_with_lag(X_takens, visits, fill_score, fill_flare)
        pred_takens_lag_reg, _ = self._fit_reg(X_takens_lag, y_reg, train_idx, test_idx, "ridge")
        pred_takens_lag_cls, prob_takens_lag_cls, _ = self._fit_cls(X_takens_lag, y_cls, train_idx, test_idx, "logistic")

        # Flare-onset secondary target (today=0 -> tomorrow=1). Eligible rows only.
        y_onset_raw = visits[self.cfg.target_classification_onset].astype(float).to_numpy()
        onset_valid = ~np.isnan(y_onset_raw)
        y_onset_int = np.where(onset_valid, np.nan_to_num(y_onset_raw, nan=0.0), 0).astype(int)
        onset_train = train_idx[onset_valid[train_idx]]
        onset_test = test_idx[onset_valid[test_idx]]
        flare_today_test = visits["flare_persistence_today"].astype(float).to_numpy()[onset_test]
        onset_block = self._fit_and_score_onset(
            y_onset_int[onset_train], y_onset_int[onset_test],
            X_grm[onset_train], X_grm[onset_test],
            X_grm_lag[onset_train], X_grm_lag[onset_test],
            X_raw[onset_train], X_raw[onset_test],
            flare_today_test,
            X_pca_train=X_pca[onset_train], X_pca_test=X_pca[onset_test],
            X_pca_lag_train=X_pca_lag[onset_train], X_pca_lag_test=X_pca_lag[onset_test],
            X_takens_train=X_takens[onset_train], X_takens_test=X_takens[onset_test],
            X_takens_lag_train=X_takens_lag[onset_train], X_takens_lag_test=X_takens_lag[onset_test],
        )

        self.flare_temperature = self._fit_flare_temperature(X_grm, y_cls, train_idx)
        prob_grm_cls_calibrated = self._apply_flare_temperature(self.logistic_clf, X_grm, test_idx, self.flare_temperature)
        pred_grm_cls_calibrated = (prob_grm_cls_calibrated >= 0.5).astype(int) if prob_grm_cls_calibrated is not None else None

        metrics: Dict = {
            "manifest": "model/manifest.json",
            "evaluation_tier": "transductive_diagnostic",
            "regression": {
                "grm_ridge": self._reg_metrics(y_reg[test_idx], pred_grm_reg),
                "grm_plus_lag_ridge": self._reg_metrics(y_reg[test_idx], pred_grm_lag_reg),
                "pca_ridge": self._reg_metrics(y_reg[test_idx], pred_pca_reg),
                "pca_plus_lag_ridge": self._reg_metrics(y_reg[test_idx], pred_pca_lag_reg),
                "takens_ridge": self._reg_metrics(y_reg[test_idx], pred_takens_reg),
                "takens_plus_lag_ridge": self._reg_metrics(y_reg[test_idx], pred_takens_lag_reg),
                "smooth_rbf_kernel_ridge": self._reg_metrics(y_reg[test_idx], pred_smooth_reg),
                "raw_random_forest": self._reg_metrics(y_reg[test_idx], pred_raw_reg),
                "naive_current_score": self._reg_metrics(y_reg[test_idx], pred_naive_reg),
                "persistence_yesterday_score": self._reg_metrics(y_reg[test_idx], persistence["score_pred"]),
            },
            "classification": {
                "grm_logistic": self._cls_metrics(y_cls[test_idx], pred_grm_cls, prob_grm_cls),
                "grm_logistic_calibrated": (
                    self._cls_metrics(y_cls[test_idx], pred_grm_cls_calibrated, prob_grm_cls_calibrated)
                    if prob_grm_cls_calibrated is not None else {}
                ),
                "grm_plus_lag_logistic": self._cls_metrics(y_cls[test_idx], pred_grm_lag_cls, prob_grm_lag_cls),
                "pca_logistic": self._cls_metrics(y_cls[test_idx], pred_pca_cls, prob_pca_cls),
                "pca_plus_lag_logistic": self._cls_metrics(y_cls[test_idx], pred_pca_lag_cls, prob_pca_lag_cls),
                "takens_logistic": self._cls_metrics(y_cls[test_idx], pred_takens_cls, prob_takens_cls),
                "takens_plus_lag_logistic": self._cls_metrics(y_cls[test_idx], pred_takens_lag_cls, prob_takens_lag_cls),
                "smooth_rbf_kernel_ridge": self._cls_metrics(
                    y_cls[test_idx], (prob_smooth_cls >= 0.5).astype(int), prob_smooth_cls
                ),
                "raw_random_forest": self._cls_metrics(y_cls[test_idx], pred_raw_cls, prob_raw_cls),
                "naive_current_score": self._cls_metrics(y_cls[test_idx], pred_naive_cls, prob_naive_cls),
                "persistence_yesterday_flare": self._cls_metrics(
                    y_cls[test_idx], persistence["flare_pred"], persistence["flare_prob"]
                ),
            },
            "flare_onset_classification": onset_block,
            "aliased_subset_evaluation": self._evaluate_aliased_subset(
                visits.iloc[test_idx], y_reg[test_idx], y_cls[test_idx],
                pred_grm_lag_reg, prob_grm_lag_cls,
                pred_raw_reg, prob_raw_cls,
                pred_naive_reg, prob_naive_cls,
                persistence,
                pred_pca_lag_reg=pred_pca_lag_reg,
                prob_pca_lag_cls=prob_pca_lag_cls,
                pred_takens_lag_reg=pred_takens_lag_reg,
                prob_takens_lag_cls=prob_takens_lag_cls,
            ),
            "constitution_recovery": self._evaluate_constitution_recovery(
                visits.iloc[train_idx], embeddings[train_idx],
                visits.iloc[test_idx], embeddings[test_idx],
            ),
            "spectral_signal_concentration": self._spectral_signal_concentration(visits.iloc[test_idx], embeddings[test_idx]),
            "parsimony": self._parsimony_summary(),
            "flare_temperature": float(self.flare_temperature) if self.flare_temperature is not None else None,
            "latent_recovery": self._latent_recovery_capture(visits, embeddings, latent) if latent is not None else {},
            "tier_note": (
                "Transductive diagnostic: train/test split is within the same visit graph. Eigenvectors and "
                "embeddings see ALL visits during decomposition. Held-out metrics here are upper bounds; for "
                "honest deployable numbers run grm_tcm_train.py --inductive."
            ),
        }

        pred_df = visits[["visit_id", "subject_id", "day", self.cfg.target_regression, self.cfg.target_classification]].copy()
        pred_df["split"] = "train"
        pred_df.loc[test_idx, "split"] = "test"
        pred_df["pred_grm_next_score"] = np.nan
        pred_df.loc[test_idx, "pred_grm_next_score"] = pred_grm_reg
        pred_df["pred_raw_next_score"] = np.nan
        pred_df.loc[test_idx, "pred_raw_next_score"] = pred_raw_reg
        pred_df["pred_grm_flare_prob"] = np.nan
        pred_df.loc[test_idx, "pred_grm_flare_prob"] = prob_grm_cls
        if prob_grm_cls_calibrated is not None:
            pred_df["pred_grm_flare_prob_calibrated"] = np.nan
            pred_df.loc[test_idx, "pred_grm_flare_prob_calibrated"] = prob_grm_cls_calibrated
        pred_df["pred_raw_flare_prob"] = np.nan
        pred_df.loc[test_idx, "pred_raw_flare_prob"] = prob_raw_cls
        pred_df["pred_persistence_next_score"] = np.nan
        pred_df.loc[test_idx, "pred_persistence_next_score"] = persistence["score_pred"]
        pred_df["pred_persistence_flare_prob"] = np.nan
        pred_df.loc[test_idx, "pred_persistence_flare_prob"] = persistence["flare_prob"]
        pred_df["pred_grm_plus_lag_next_score"] = np.nan
        pred_df.loc[test_idx, "pred_grm_plus_lag_next_score"] = pred_grm_lag_reg
        pred_df["pred_grm_plus_lag_flare_prob"] = np.nan
        pred_df.loc[test_idx, "pred_grm_plus_lag_flare_prob"] = prob_grm_lag_cls
        return metrics, pred_df

    def _fit_and_score_onset(
        self,
        y_onset_train: np.ndarray, y_onset_test: np.ndarray,
        X_grm_train: np.ndarray, X_grm_test: np.ndarray,
        X_grm_lag_train: np.ndarray, X_grm_lag_test: np.ndarray,
        X_raw_train: np.ndarray, X_raw_test: np.ndarray,
        flare_today_test: np.ndarray,
        *,
        X_pca_train: Optional[np.ndarray] = None,
        X_pca_test: Optional[np.ndarray] = None,
        X_pca_lag_train: Optional[np.ndarray] = None,
        X_pca_lag_test: Optional[np.ndarray] = None,
        X_takens_train: Optional[np.ndarray] = None,
        X_takens_test: Optional[np.ndarray] = None,
        X_takens_lag_train: Optional[np.ndarray] = None,
        X_takens_lag_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fit flare_onset classifiers and score on the pre-sliced eligible subset.

        Returns metrics on both the full eligible set AND the hard subset
        (flare_today == 0). The hard subset closes the inflation loophole:
        on the full set, any classifier that uses flare_today can trivially
        predict 0 whenever flare_today=1 (definitionally certain), inflating AUC.
        The hard subset removes those guaranteed-zero rows so the reported AUC
        reflects genuine onset prediction, not the trivial filter.
        """

        if len(np.unique(y_onset_train)) < 2 or len(y_onset_test) == 0:
            return {}

        clf_grm = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_grm_train, y_onset_train)
        prob_grm = clf_grm.predict_proba(X_grm_test)[:, 1]
        pred_grm = (prob_grm >= 0.5).astype(int)

        clf_grm_lag = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_grm_lag_train, y_onset_train)
        prob_grm_lag = clf_grm_lag.predict_proba(X_grm_lag_test)[:, 1]
        pred_grm_lag = (prob_grm_lag >= 0.5).astype(int)

        # PCA baselines (when provided).
        prob_pca: Optional[np.ndarray] = None
        pred_pca: Optional[np.ndarray] = None
        prob_pca_lag: Optional[np.ndarray] = None
        pred_pca_lag: Optional[np.ndarray] = None
        if X_pca_train is not None and X_pca_test is not None:
            clf_pca = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_pca_train, y_onset_train)
            prob_pca = clf_pca.predict_proba(X_pca_test)[:, 1]
            pred_pca = (prob_pca >= 0.5).astype(int)
        if X_pca_lag_train is not None and X_pca_lag_test is not None:
            clf_pca_lag = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_pca_lag_train, y_onset_train)
            prob_pca_lag = clf_pca_lag.predict_proba(X_pca_lag_test)[:, 1]
            pred_pca_lag = (prob_pca_lag >= 0.5).astype(int)

        # Delay-embedded (Takens) baselines (when provided).
        prob_takens: Optional[np.ndarray] = None
        pred_takens: Optional[np.ndarray] = None
        prob_takens_lag: Optional[np.ndarray] = None
        pred_takens_lag: Optional[np.ndarray] = None
        if X_takens_train is not None and X_takens_test is not None:
            clf_tak = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_takens_train, y_onset_train)
            prob_takens = clf_tak.predict_proba(X_takens_test)[:, 1]
            pred_takens = (prob_takens >= 0.5).astype(int)
        if X_takens_lag_train is not None and X_takens_lag_test is not None:
            clf_tak_lag = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_takens_lag_train, y_onset_train)
            prob_takens_lag = clf_tak_lag.predict_proba(X_takens_lag_test)[:, 1]
            pred_takens_lag = (prob_takens_lag >= 0.5).astype(int)

        # Lag-only baseline: the last two columns of X_grm_lag are the persistence lags.
        # This is the "trivial filter" baseline that bounds how much of the headline AUC
        # comes from the flare_today==1 -> onset=0 certainty alone.
        X_lag_only_train = X_grm_lag_train[:, -2:]
        X_lag_only_test = X_grm_lag_test[:, -2:]
        clf_lag_only = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_lag_only_train, y_onset_train)
        prob_lag_only = clf_lag_only.predict_proba(X_lag_only_test)[:, 1]
        pred_lag_only = (prob_lag_only >= 0.5).astype(int)

        clf_raw = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=4, random_state=42, n_jobs=-1, class_weight="balanced",
        ).fit(X_raw_train, y_onset_train)
        prob_raw = clf_raw.predict_proba(X_raw_test)[:, 1]
        pred_raw = (prob_raw >= 0.5).astype(int)

        onset_marginal = float(np.mean(y_onset_train))
        prob_naive = np.full(len(y_onset_test), onset_marginal)
        pred_naive = (prob_naive >= 0.5).astype(int)

        def _block(mask: np.ndarray) -> Dict[str, Any]:
            yt = y_onset_test[mask]
            block: Dict[str, Any] = {
                "n_test_eligible": int(mask.sum()),
                "n_test_positive": int(yt.sum()),
            }
            if len(np.unique(yt)) < 2 or mask.sum() == 0:
                return block
            block["grm_logistic"] = self._cls_metrics(yt, pred_grm[mask], prob_grm[mask])
            block["grm_plus_lag_logistic"] = self._cls_metrics(yt, pred_grm_lag[mask], prob_grm_lag[mask])
            if pred_pca is not None:
                block["pca_logistic"] = self._cls_metrics(yt, pred_pca[mask], prob_pca[mask])
            if pred_pca_lag is not None:
                block["pca_plus_lag_logistic"] = self._cls_metrics(yt, pred_pca_lag[mask], prob_pca_lag[mask])
            if pred_takens is not None:
                block["takens_logistic"] = self._cls_metrics(yt, pred_takens[mask], prob_takens[mask])
            if pred_takens_lag is not None:
                block["takens_plus_lag_logistic"] = self._cls_metrics(yt, pred_takens_lag[mask], prob_takens_lag[mask])
            block["lag_only_logistic"] = self._cls_metrics(yt, pred_lag_only[mask], prob_lag_only[mask])
            block["raw_random_forest"] = self._cls_metrics(yt, pred_raw[mask], prob_raw[mask])
            block["naive_marginal"] = self._cls_metrics(yt, pred_naive[mask], prob_naive[mask])
            return block

        full_mask = np.ones(len(y_onset_test), dtype=bool)
        hard_mask = (flare_today_test == 0)

        out = _block(full_mask)
        out["train_marginal"] = onset_marginal
        out["hard_subset_flare_today_0"] = _block(hard_mask)
        out["hard_subset_flare_today_0"]["note"] = (
            "AUC restricted to rows where flare_today=0. Removes the trivial filter "
            "(flare_today=1 -> onset=0 certainty) so the metric reflects real onset prediction."
        )
        return out

    @staticmethod
    def _augment_with_lag(
        X_grm: np.ndarray, visits: pd.DataFrame,
        fill_score: float, fill_flare: float,
    ) -> np.ndarray:
        """Concatenate per-subject target lags onto the GRM embedding matrix.

        Yields the feature matrix used by the "GRM + lag" head. The lag columns are
        the same persistence signals fed to the persistence baseline, so the head
        AT WORST recovers persistence performance, and at best improves on it.
        First-visit-per-subject NaNs are filled with the supplied train marginals
        (not test marginals — caller's responsibility).
        """

        lag_s = visits["score_persistence_today"].astype(float).to_numpy()
        lag_s = np.where(np.isnan(lag_s), fill_score, lag_s)
        lag_f = visits["flare_persistence_today"].astype(float).to_numpy()
        lag_f = np.where(np.isnan(lag_f), fill_flare, lag_f)
        return np.column_stack([X_grm, lag_s.reshape(-1, 1), lag_f.reshape(-1, 1)])

    @staticmethod
    def _build_delay_embedding(
        X: np.ndarray, visits: pd.DataFrame, k: int,
    ) -> np.ndarray:
        """Concatenate last k visits into a single feature vector per row (Takens embedding).

        Respects subject boundaries: the first visit of each subject pads missing
        history with NaN, which is then median-imputed. This ensures no cross-subject
        leakage and no future leakage (only past observations are used).

        Returns X_takens of shape (N, p * k) where p = X.shape[1].
        """

        if k <= 1:
            return X.copy()
        p = X.shape[1]
        n = X.shape[0]
        subject_ids = visits["subject_id"].to_numpy(int)
        # Pre-fill with NaN so early visits get imputed.
        X_takens = np.full((n, p * k), np.nan, dtype=float)
        X_takens[:, :p] = X  # lag-0 = current visit
        for lag in range(1, k):
            shifted = np.full((n, p), np.nan, dtype=float)
            # Vectorized per-subject shift: compare subject_id at row i and row i-lag.
            if lag <= n - 1:
                valid = np.zeros(n, dtype=bool)
                valid[lag:] = subject_ids[lag:] == subject_ids[:-lag]
                shifted[valid] = X[np.where(valid)[0] - lag]
            X_takens[:, lag * p:(lag + 1) * p] = shifted
        # Median-impute NaN columns (early visits per subject).
        col_medians = np.nanmedian(X_takens, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        nan_mask = np.isnan(X_takens)
        X_takens[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        return X_takens

    @staticmethod
    def _persistence_baseline(
        eval_visits: pd.DataFrame,
        train_visits: pd.DataFrame,
        target_classification: str,
        target_regression: str,
    ) -> Dict[str, np.ndarray]:
        """Per-subject persistence baselines: yesterday's label predicts today's label.

        Returns dict with keys:
          - 'flare_pred', 'flare_prob': binary persistence for `target_classification`.
            First-visit fallback is the train-set marginal flare rate.
          - 'score_pred': regression persistence for `target_regression`.
            First-visit fallback is the train-set median.
        """

        train_flare_rate = float(np.nanmean(train_visits[target_classification].astype(float).to_numpy()))
        train_score_median = float(np.nanmedian(train_visits[target_regression].astype(float).to_numpy()))

        flare_prob = eval_visits["flare_persistence_today"].astype(float).to_numpy()
        flare_prob = np.where(np.isnan(flare_prob), train_flare_rate, flare_prob)
        flare_pred = (flare_prob >= 0.5).astype(int)

        score_pred = eval_visits["score_persistence_today"].astype(float).to_numpy()
        score_pred = np.where(np.isnan(score_pred), train_score_median, score_pred)
        return {
            "flare_pred": flare_pred,
            "flare_prob": flare_prob,
            "score_pred": score_pred,
        }

    def _raw_baseline_matrix(self, visits: pd.DataFrame) -> np.ndarray:
        cols = (self.feature_names or list(OBSERVATION_NAMES)) + ["global_dysregulation_score"]
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        return pipe.fit_transform(visits[cols].to_numpy(float))

    @staticmethod
    def _fit_reg(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, model: str) -> Tuple[np.ndarray, BaseEstimator]:
        if model == "ridge":
            reg = Ridge(alpha=1.0)
        elif model == "random_forest":
            reg = RandomForestRegressor(n_estimators=100, min_samples_leaf=4, random_state=42, n_jobs=-1)
        else:
            raise ValueError(model)
        reg.fit(X[train_idx], y[train_idx])
        return reg.predict(X[test_idx]), reg

    @staticmethod
    def _fit_smooth_rbf_baseline(
        X_train: np.ndarray,
        y_reg_train: np.ndarray,
        y_cls_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Observation-only smooth distance-kernel baseline.

        This is the Pang-EDR analogue for visits: a stripped-down RBF kernel over
        observed visit features, with no temporal edges, treatment edges, graph
        eigenmodes, or subject similarity. If this matches GRM, the load-bearing
        property is probably distance-smoothness rather than the full GRM machinery.
        """

        if len(X_train) < 2:
            return np.zeros(len(X_test), dtype=float), np.full(len(X_test), 0.5, dtype=float)
        nn = NearestNeighbors(n_neighbors=min(2, len(X_train)), metric="euclidean").fit(X_train)
        distances, _ = nn.kneighbors(X_train)
        nonzero = distances[:, 1:].ravel()
        nonzero = nonzero[nonzero > 0]
        sigma = float(np.median(nonzero) if len(nonzero) else 1.0)
        gamma = 1.0 / (2.0 * max(sigma, 1e-9) ** 2)

        reg = KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma).fit(X_train, y_reg_train)
        pred_reg = np.asarray(reg.predict(X_test), dtype=float)

        if len(np.unique(y_cls_train)) < 2:
            prob_cls = np.full(len(X_test), float(y_cls_train[0]) if len(y_cls_train) else 0.5)
        else:
            cls = KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma).fit(X_train, y_cls_train.astype(float))
            prob_cls = np.clip(np.asarray(cls.predict(X_test), dtype=float), 0.0, 1.0)
        return pred_reg, prob_cls

    @staticmethod
    def _fit_cls(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, model: str) -> Tuple[np.ndarray, np.ndarray, Optional[BaseEstimator]]:
        if len(np.unique(y[train_idx])) < 2:
            constant = int(y[train_idx][0]) if len(train_idx) else 0
            pred = np.full(len(test_idx), constant, dtype=int)
            prob = np.full(len(test_idx), float(constant), dtype=float)
            return pred, prob, None
        if model == "logistic":
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        elif model == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, random_state=42, n_jobs=-1, class_weight="balanced")
        else:
            raise ValueError(model)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        prob = clf.predict_proba(X[test_idx])[:, 1]
        return pred, prob, clf

    def _fit_flare_temperature(
        self, X: np.ndarray, y_cls: np.ndarray, train_idx: np.ndarray, n_inner_folds: int = 5,
    ) -> Optional[float]:
        """Fit a single temperature T > 0 for the flare classifier on inner-CV held-out logits.

        Why inner CV: temperature must be fit on predictions the calibrator hasn't seen
        in training, otherwise T collapses toward 1. We use KFold within train_idx so the
        outer test fold is never touched.
        """

        if len(np.unique(y_cls[train_idx])) < 2 or len(train_idx) < n_inner_folds * 2:
            return None
        kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=self.cfg.random_seed)
        logits: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for inner_tr, inner_val in kf.split(train_idx):
            sub_tr = train_idx[inner_tr]
            sub_val = train_idx[inner_val]
            if len(np.unique(y_cls[sub_tr])) < 2:
                continue
            inner_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            inner_clf.fit(X[sub_tr], y_cls[sub_tr])
            logits.append(inner_clf.decision_function(X[sub_val]))
            labels.append(y_cls[sub_val])
        if not logits:
            return None
        z = np.concatenate(logits)
        y = np.concatenate(labels)
        return _fit_binary_temperature(z, y)

    @staticmethod
    def _apply_flare_temperature(
        clf: Optional[BaseEstimator], X: np.ndarray, test_idx: np.ndarray, T: Optional[float],
    ) -> Optional[np.ndarray]:
        """Apply temperature T to a binary logistic classifier's decision_function on test_idx."""

        if clf is None or T is None or not np.isfinite(T) or T <= 0:
            return None
        z = clf.decision_function(X[test_idx])
        return _sigmoid(z / float(T))

    @staticmethod
    def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def _cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        out = {"accuracy": float(accuracy_score(y_true, y_pred))}
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        y_prob_safe = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1.0 - 1e-12)
        out["log_loss"] = float(-np.mean(np.where(y_true == 1, np.log(y_prob_safe), np.log(1.0 - y_prob_safe))))
        out["brier"] = float(np.mean((y_prob_safe - y_true) ** 2))
        return out

    def _evaluate_aliased_subset(
        self,
        test_visits: pd.DataFrame,
        y_reg_test: np.ndarray, y_cls_test: np.ndarray,
        pred_grm_lag_reg: np.ndarray, prob_grm_lag_cls: np.ndarray,
        pred_raw_reg: np.ndarray, prob_raw_cls: np.ndarray,
        pred_naive_reg: np.ndarray, prob_naive_cls: np.ndarray,
        persistence: Dict[str, np.ndarray],
        *,
        pred_pca_lag_reg: Optional[np.ndarray] = None,
        prob_pca_lag_cls: Optional[np.ndarray] = None,
        pred_takens_lag_reg: Optional[np.ndarray] = None,
        prob_takens_lag_cls: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Score predictions on the aliased-pair subset only.

        Aliased-pair rows: today's observations sit in an alias group with 2+
        regimes (e.g., stressed_recoverable vs stuck_agitated both look "wired"),
        but next-regime distributions diverge sharply. With constitution_dynamics
        enabled, the divergence is subject-conditional: K determines which member
        of the aliased pair you are in. Methods that exploit graph position
        (regime trajectory) should hold up; pure-obs and lag-only baselines lose
        the most ground here.
        """

        if "is_aliased_pair_row" not in test_visits.columns:
            return {}
        mask = (test_visits["is_aliased_pair_row"].to_numpy() == 1)
        if mask.sum() < 10:
            return {}

        out: Dict[str, Any] = {
            "n_eligible": int(mask.sum()),
            "regression": {
                "grm_plus_lag_ridge": self._reg_metrics(y_reg_test[mask], pred_grm_lag_reg[mask]),
                "raw_random_forest": self._reg_metrics(y_reg_test[mask], pred_raw_reg[mask]),
                "naive_current_score": self._reg_metrics(y_reg_test[mask], pred_naive_reg[mask]),
                "persistence_yesterday_score": self._reg_metrics(y_reg_test[mask], persistence["score_pred"][mask]),
            },
            "note": (
                "Aliased-pair subset (observations alias across regimes, futures diverge). "
                "Graph-position-aware methods should beat pure-obs and lag-only here when "
                "constitution_dynamics_strength > 0."
            ),
        }
        if pred_pca_lag_reg is not None:
            out["regression"]["pca_plus_lag_ridge"] = self._reg_metrics(y_reg_test[mask], pred_pca_lag_reg[mask])
        if pred_takens_lag_reg is not None:
            out["regression"]["takens_plus_lag_ridge"] = self._reg_metrics(y_reg_test[mask], pred_takens_lag_reg[mask])
        if len(np.unique(y_cls_test[mask])) >= 2:
            out["classification"] = {
                "grm_plus_lag_logistic": self._cls_metrics(
                    y_cls_test[mask], (prob_grm_lag_cls[mask] >= 0.5).astype(int), prob_grm_lag_cls[mask]
                ),
                "raw_random_forest": self._cls_metrics(
                    y_cls_test[mask], (prob_raw_cls[mask] >= 0.5).astype(int), prob_raw_cls[mask]
                ),
                "naive_current_score": self._cls_metrics(
                    y_cls_test[mask], (prob_naive_cls[mask] >= 0.5).astype(int), prob_naive_cls[mask]
                ),
                "persistence_yesterday_flare": self._cls_metrics(
                    y_cls_test[mask],
                    np.asarray(persistence["flare_pred"])[mask],
                    np.asarray(persistence["flare_prob"])[mask],
                ),
            }
            if prob_pca_lag_cls is not None:
                out["classification"]["pca_plus_lag_logistic"] = self._cls_metrics(
                    y_cls_test[mask], (prob_pca_lag_cls[mask] >= 0.5).astype(int), prob_pca_lag_cls[mask]
                )
            if prob_takens_lag_cls is not None:
                out["classification"]["takens_plus_lag_logistic"] = self._cls_metrics(
                    y_cls_test[mask], (prob_takens_lag_cls[mask] >= 0.5).astype(int), prob_takens_lag_cls[mask]
                )
        return out

    def _evaluate_constitution_recovery(
        self,
        train_visits: pd.DataFrame, train_embeddings: np.ndarray,
        test_visits: pd.DataFrame, test_embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        """Constitution-recovery evaluation: predict per-subject constitution axes
        from per-subject mean of GRM embeddings vs per-subject mean of raw features.

        Held-out by subject: train Ridge on TRAIN subjects' aggregates, score on
        TEST subjects' aggregates. The claim under test: does GRM aggregation
        recover the stable cross-modal constitution layer better than naive
        per-subject averaging of the same input features?

        Skipped when subjects.csv is missing constitution columns. Returns {} in
        that case so callers can no-op.
        """

        if not self.cfg.evaluate_constitution_recovery or self.subjects is None:
            return {}
        const_cols = [c for c in CONSTITUTION_NAMES if c in self.subjects.columns]
        if not const_cols:
            return {}

        # Per-subject aggregates: mean of features and mean of GRM coordinates.
        # Mean is the simplest aggregation and the natural fit for a *stable*
        # subject-level target. (Variance or first-eigenmode aggregates are
        # natural follow-ups for nonstationary targets — not needed here.)
        feature_cols = self.feature_names or list(OBSERVATION_NAMES)

        def _aggregate(visits: pd.DataFrame, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            # Align embeddings with visits by row order; both are sorted by subject_id, day.
            sub_ids = visits["subject_id"].to_numpy(int)
            unique_subs = np.array(sorted(np.unique(sub_ids)))
            emb_means = np.zeros((len(unique_subs), embeddings.shape[1]))
            raw_means = np.zeros((len(unique_subs), len(feature_cols)))
            X_raw = visits[feature_cols].to_numpy(float)
            X_raw = np.where(np.isnan(X_raw), np.nanmedian(X_raw, axis=0), X_raw)
            for i, s in enumerate(unique_subs):
                mask = sub_ids == int(s)
                emb_means[i] = embeddings[mask].mean(axis=0)
                raw_means[i] = X_raw[mask].mean(axis=0)
            const_for_sub = self.subjects.set_index("subject_id").reindex(unique_subs)[const_cols].to_numpy(float)
            return unique_subs, emb_means, raw_means, const_for_sub

        tr_sub, tr_emb_mean, tr_raw_mean, tr_y = _aggregate(train_visits, train_embeddings)
        te_sub, te_emb_mean, te_raw_mean, te_y = _aggregate(test_visits, test_embeddings)

        if len(tr_sub) < 3 or len(te_sub) < 2:
            return {}

        results: Dict[str, Any] = {
            "n_train_subjects": int(len(tr_sub)),
            "n_test_subjects": int(len(te_sub)),
            "axes": const_cols,
            "note": (
                "Per-subject mean(GRM embedding) -> constitution axes via Ridge, scored on "
                "held-out subjects. Baselines: per-subject mean(raw features) -> constitution. "
                "Constitution is a STABLE subject-level target; mean aggregation is the natural reduction."
            ),
            "grm_aggregate_ridge": {},
            "subject_graph_grm_ridge": {},
            "raw_aggregate_ridge": {},
        }

        def _fit_predict(X_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
            scaler = StandardScaler().fit(X_tr)
            Xt = scaler.transform(X_tr)
            Xv = scaler.transform(X_te)
            preds = np.zeros((X_te.shape[0], len(const_cols)))
            for j in range(len(const_cols)):
                ridge = Ridge(alpha=1.0).fit(Xt, tr_y[:, j])
                preds[:, j] = ridge.predict(Xv)
            return preds

        grm_pred = _fit_predict(tr_emb_mean, te_emb_mean)
        subject_grm_train, subject_grm_test = self._subject_graph_grm_features(tr_raw_mean, te_raw_mean)
        subject_grm_pred = _fit_predict(subject_grm_train, subject_grm_test)
        raw_pred = _fit_predict(tr_raw_mean, te_raw_mean)

        for j, axis in enumerate(const_cols):
            results["grm_aggregate_ridge"][axis] = self._reg_metrics(te_y[:, j], grm_pred[:, j])
            results["subject_graph_grm_ridge"][axis] = self._reg_metrics(te_y[:, j], subject_grm_pred[:, j])
            results["raw_aggregate_ridge"][axis] = self._reg_metrics(te_y[:, j], raw_pred[:, j])

        results["mean_r2"] = {
            "grm_aggregate_ridge": float(np.mean([results["grm_aggregate_ridge"][a]["r2"] for a in const_cols])),
            "subject_graph_grm_ridge": float(np.mean([results["subject_graph_grm_ridge"][a]["r2"] for a in const_cols])),
            "raw_aggregate_ridge": float(np.mean([results["raw_aggregate_ridge"][a]["r2"] for a in const_cols])),
        }
        return results

    def _subject_graph_grm_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Subject-level GRM features for stable constitution recovery.

        This is the architectural counterpoint to mean(visit embeddings): first
        collapse each subject to a stable aggregate, then build the graph over
        subjects. Test subjects are Nyström-projected from their aggregate into
        the train-subject graph.
        """

        X_train = np.asarray(X_train, dtype=float)
        X_test = np.asarray(X_test, dtype=float)
        n_train = X_train.shape[0]
        if n_train < 4:
            return X_train, X_test

        imputer = SimpleImputer(strategy="median").fit(X_train)
        X_tr = imputer.transform(X_train)
        X_te = imputer.transform(X_test)
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_te = scaler.transform(X_te)

        n_neighbors = min(max(int(self.cfg.subject_similarity_neighbors), 1) + 1, n_train)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X_tr)
        distances, indices = nn.kneighbors(X_tr)
        nonzero = distances[:, 1:].ravel()
        nonzero = nonzero[nonzero > 0]
        sigma = float(np.median(nonzero) if len(nonzero) else 1.0)
        sigma = max(sigma, 1e-9)

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        for i in range(n_train):
            for dist, j in zip(distances[i, 1:], indices[i, 1:]):
                weight = float(np.exp(-(float(dist) ** 2) / (2.0 * sigma ** 2)))
                rows += [i, int(j)]
                cols += [int(j), i]
                vals += [weight, weight]
        W = sparse.coo_matrix((vals, (rows, cols)), shape=(n_train, n_train)).tocsr()
        W.setdiag(0.0)
        W.eliminate_zeros()
        W = W.maximum(W.T)

        degrees = np.maximum(np.asarray(W.sum(axis=1)).ravel(), 1e-12)
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
        L = sparse.eye(n_train, format="csr") - D_inv_sqrt @ W @ D_inv_sqrt
        k = min(max(int(self.cfg.n_modes), 1) + 1, n_train - 2)
        if k < 2:
            return X_tr, X_te
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order][1:self.cfg.n_modes + 1]
        eigenvectors = canonicalize_eigvec_signs(eigenvectors[:, order][:, 1:self.cfg.n_modes + 1])
        # Diffusion-map convention (matches _make_grm_embeddings): sqrt-weighted
        # so embedding inner products literally reconstruct the GRM kernel.
        weights = 1.0 / np.sqrt(1.0 + (self.cfg.rho ** 2) * eigenvalues)
        train_features = eigenvectors * weights.reshape(1, -1)

        test_features = nystrom_extend_arrays(
            X_te,
            nn_index=nn,
            knn_sigma=sigma,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            rho=self.cfg.rho,
            normalized=True,
            train_degrees=degrees,
            n_neighbors=max(1, n_neighbors - 1),
        )
        return train_features, test_features

    def _spectral_signal_concentration(self, visits: pd.DataFrame, embeddings: np.ndarray) -> Dict[str, Any]:
        """Report how concentrated target/regime signal is in early GRM modes.

        Pang-style analogue of "activity lives at long wavelengths": for each
        target, compute per-mode squared association and ask how many low-index
        modes carry 75% of the total association.
        """

        if embeddings.size == 0:
            return {}

        def _mode_signal(y: np.ndarray) -> Dict[str, Any]:
            y = np.asarray(y, dtype=float)
            mask = np.isfinite(y)
            if mask.sum() < 3 or np.nanstd(y[mask]) <= 1e-12:
                return {}
            scores = []
            for j in range(embeddings.shape[1]):
                x = embeddings[:, j]
                if np.nanstd(x[mask]) <= 1e-12:
                    scores.append(0.0)
                else:
                    corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
                    scores.append(0.0 if not np.isfinite(corr) else corr * corr)
            scores_arr = np.asarray(scores, dtype=float)
            total = float(scores_arr.sum())
            if total <= 1e-12:
                return {"modes_for_75pct_signal": None, "early_4_signal_fraction": 0.0}
            cumulative = np.cumsum(scores_arr) / total
            modes_75 = int(np.searchsorted(cumulative, 0.75) + 1)
            return {
                "modes_for_75pct_signal": modes_75,
                "early_4_signal_fraction": float(cumulative[min(3, len(cumulative) - 1)]),
                "per_mode_signal_fraction": [float(v / total) for v in scores_arr],
            }

        out: Dict[str, Any] = {}
        if self.cfg.target_regression in visits.columns:
            out[self.cfg.target_regression] = _mode_signal(visits[self.cfg.target_regression].to_numpy(float))
        if self.cfg.target_classification in visits.columns:
            out[self.cfg.target_classification] = _mode_signal(visits[self.cfg.target_classification].astype(float).to_numpy())
        if "true_regime_id" in visits.columns:
            regime_scores = np.zeros(embeddings.shape[1], dtype=float)
            regimes = visits["true_regime_id"].astype(float).to_numpy()
            for rid in sorted(pd.Series(regimes).dropna().unique()):
                block = _mode_signal((regimes == rid).astype(float))
                frac = np.asarray(block.get("per_mode_signal_fraction", []), dtype=float)
                if frac.size == embeddings.shape[1]:
                    regime_scores += frac
            total = float(regime_scores.sum())
            if total > 1e-12:
                cumulative = np.cumsum(regime_scores) / total
                out["true_regime_id"] = {
                    "modes_for_75pct_signal": int(np.searchsorted(cumulative, 0.75) + 1),
                    "early_4_signal_fraction": float(cumulative[min(3, len(cumulative) - 1)]),
                    "per_mode_signal_fraction": [float(v / total) for v in regime_scores],
                }
        return out

    def _parsimony_summary(self) -> Dict[str, Any]:
        """Fixed hyperparameter counts for model-family comparison."""

        return {
            "note": (
                "Counts are configured structural knobs, not fitted coefficient counts. "
                "They make the parsimony tradeoff explicit beside predictive metrics."
            ),
            "grm_structural_hyperparameters": {
                "count": 9,
                "items": [
                    "n_modes", "rho", "n_neighbors", "similarity_sigma",
                    "diffusion_alpha", "temporal_edge_weight", "same_subject_edge_weight",
                    "treatment_edge_weight", "graph_mode",
                ],
            },
            "grm_plus_lag_extra_hyperparameters": {"count": 1, "items": ["use_lag_features"]},
            "smooth_rbf_kernel_ridge": {"count": 2, "items": ["rbf_sigma_median_heuristic", "ridge_alpha"]},
            "raw_random_forest": {"count": 4, "items": ["n_estimators", "min_samples_leaf", "class_weight", "random_seed"]},
            "persistence_yesterday": {"count": 0, "items": []},
        }

    def _latent_recovery_capture(self, visits: pd.DataFrame, embeddings: np.ndarray, latent: pd.DataFrame) -> Dict[str, object]:
        merged = visits[["visit_id", "subject_id", "day"]].merge(
            latent[["subject_id", "day"] + LATENT_NAMES], on=["subject_id", "day"], how="left"
        )
        Z = merged[LATENT_NAMES].to_numpy(float)
        Z = SimpleImputer(strategy="median").fit_transform(Z)
        Z = StandardScaler().fit_transform(Z)
        E = StandardScaler().fit_transform(embeddings)
        q = min(E.shape[1], Z.shape[1])
        R, _ = orthogonal_procrustes(E[:, :q], Z[:, :q])
        self.procrustes_R = R
        E_aligned = E[:, :q] @ R
        corrs = [float(np.corrcoef(E_aligned[:, j], Z[:, j])[0, 1]) for j in range(q)]
        return {
            "mean_abs_aligned_correlation": float(np.nanmean(np.abs(corrs))),
            "aligned_correlations": corrs,
            "note": "Synthetic-only metric: GRM embedding aligned to true latent states by orthogonal Procrustes.",
        }

    @staticmethod
    def _feature_mode_correlations(visits: pd.DataFrame, embeddings: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        X = visits[feature_names].copy()
        X = SimpleImputer(strategy="median").fit_transform(X)
        X = StandardScaler().fit_transform(X)
        rows = []
        for m in range(embeddings.shape[1]):
            mode = embeddings[:, m]
            for j, feat in enumerate(feature_names):
                corr = float(np.corrcoef(mode, X[:, j])[0, 1])
                rows.append({"mode": f"grm_mode_{m + 1}", "feature": feat, "correlation": corr, "abs_correlation": abs(corr)})
        return pd.DataFrame(rows).sort_values(["mode", "abs_correlation"], ascending=[True, False])

    def _fit_embedding_surrogate(self, X_obs: np.ndarray, embeddings: np.ndarray) -> None:
        """Fit a Ridge regressor X_obs -> embeddings on the training split.

        The surrogate is the default projection used by predict.py: at inference
        we feed standardized observations through it and treat the output as the
        GRM coordinates. This is faithful for downstream heads that consume the
        embeddings as features, and avoids the structural inaccuracy of feature-only
        Nyström extension on the multi-relational training graph. Fit on train_idx
        only so the surrogate carries the same train/test discipline as the heads.
        """

        if self.train_idx is None or len(self.train_idx) == 0:
            return
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(X_obs[self.train_idx], embeddings[self.train_idx])
        self.embedding_surrogate = surrogate

    def _write_outputs(self, embeddings_df: pd.DataFrame, feature_modes: pd.DataFrame, metrics: Dict, predictions: pd.DataFrame) -> None:
        embeddings_df.to_csv(self.output_dir / "grm_visit_embeddings.csv", index=False)
        feature_modes.to_csv(self.output_dir / "grm_feature_modes.csv", index=False)
        predictions.to_csv(self.output_dir / "grm_predictions.csv", index=False)
        with open(self.output_dir / "grm_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def _save_model(self, manifest_extra: Optional[Dict[str, Any]] = None) -> None:
        """Persist the fitted static GRM-TCM pipeline."""

        if self.eigenvalues is None or self.eigenvectors is None:
            raise RuntimeError("Model state missing; run() must populate eigenpairs before _save_model().")
        model_dir = self.output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.obs_preprocessor is not None:
            save_joblib(self.obs_preprocessor, model_dir / "obs_preprocessor.joblib")

        basis_arrays = dict(
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            rho=np.asarray(self.cfg.rho, dtype=float),
            normalized=np.asarray(self.cfg.use_normalized_laplacian, dtype=bool),
            n_modes=np.asarray(self.cfg.n_modes, dtype=int),
            graph_mode=np.asarray(self.cfg.graph_mode),
        )
        if self.train_degrees is not None:
            basis_arrays["train_degrees"] = self.train_degrees
        if self.eigenvalues_full is not None:
            basis_arrays["eigenvalues_full"] = self.eigenvalues_full
        np.savez_compressed(model_dir / "grm_basis.npz", **basis_arrays)

        if self.nn_index is not None:
            save_joblib(self.nn_index, model_dir / "nn_index.joblib")
            with open(model_dir / "knn_sigma.json", "w", encoding="utf-8") as f:
                json.dump({"knn_sigma": float(self.knn_sigma) if self.knn_sigma is not None else None}, f)

        if self.ridge_reg is not None:
            save_joblib(self.ridge_reg, model_dir / "ridge_next_day.joblib")
        if self.logistic_clf is not None:
            save_joblib(self.logistic_clf, model_dir / "logistic_flare.joblib")
        if self.embedding_surrogate is not None:
            save_joblib(self.embedding_surrogate, model_dir / "embedding_surrogate.joblib")
        if self.flare_temperature is not None:
            with open(model_dir / "flare_temperature.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "T": float(self.flare_temperature),
                        "fit_method": "inner_5fold_cv_on_train",
                        "applies_to": "logistic_flare",
                    },
                    f,
                    indent=2,
                )

        if self.train_idx is not None and self.test_idx is not None:
            with open(model_dir / "split_indices.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "seed": int(self.cfg.random_seed),
                        "test_size": float(self.cfg.test_size),
                        "train": [int(i) for i in self.train_idx],
                        "test": [int(i) for i in self.test_idx],
                    },
                    f,
                )

        if self.procrustes_R is not None:
            np.save(model_dir / "procrustes_R.npy", self.procrustes_R)

        if self._visit_index is not None:
            self._visit_index.to_parquet(model_dir / "visit_index.parquet", index=False)

        manifest_extra_full = dict(manifest_extra or {})
        manifest_extra_full["embedding_convention"] = "sqrt_grm_kernel_feature"
        if self.feature_names is not None:
            manifest_extra_full["feature_names"] = list(self.feature_names)
        write_manifest(
            model_dir,
            config=self.cfg,
            inputs=[
                self.input_dir / "visits.csv",
                self.input_dir / "latent_states.csv",
                self.input_dir / "events.csv",
            ],
            schema_version=STATIC_SCHEMA_VERSION,
            random_seed=self.cfg.random_seed,
            extra=manifest_extra_full,
        )

    def _write_inductive_plots(self, metrics: Dict[str, Any], pred_df: pd.DataFrame) -> None:
        """Write the Phase 1 headline plots for an inductive run.

        Three plots are produced; each is optional and skipped (with a clear print)
        when its prerequisite is missing:
          - transductive_vs_inductive_metrics: needs the sibling transductive
            grm_metrics.json. Path is read from cfg.transductive_results_dir.
          - flare_calibration_raw_vs_temperature: needs both raw and calibrated
            flare probabilities in pred_df.
          - per_subject_performance: needs flare_next_day + next_day_score in pred_df.
        """

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from grm_tcm_plot_captions import save_with_caption

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # ----- Plot 1: transductive vs inductive metrics -----
        transductive_path = Path(self.cfg.transductive_results_dir) / "grm_metrics.json"
        if transductive_path.exists():
            with open(transductive_path, "r", encoding="utf-8") as f:
                tmetrics = json.load(f)
            try:
                self._plot_transductive_vs_inductive(tmetrics, metrics, plot_dir, save_with_caption, plt)
            except Exception as exc:  # plot_dir is recoverable; don't blow up the run
                print(f"[plot] skipped transductive_vs_inductive_metrics: {exc}")
        else:
            print(
                f"[plot] transductive metrics not found at {transductive_path}; "
                f"skipping transductive_vs_inductive_metrics plot."
            )

        # ----- Plot 3: flare calibration raw vs temperature -----
        has_raw = "pred_grm_flare_prob" in pred_df.columns
        has_cal = "pred_grm_flare_prob_calibrated" in pred_df.columns
        target_cls = self.cfg.target_classification
        if has_raw and has_cal and target_cls in pred_df.columns:
            try:
                self._plot_flare_calibration(pred_df, plot_dir, save_with_caption, plt)
            except Exception as exc:
                print(f"[plot] skipped flare_calibration_raw_vs_temperature: {exc}")
        else:
            print("[plot] flare calibration columns missing; skipping calibration plot.")

        # ----- Plot 4: per-subject performance -----
        try:
            self._plot_per_subject_performance(pred_df, plot_dir, save_with_caption, plt)
        except Exception as exc:
            print(f"[plot] skipped per_subject_performance: {exc}")

    @staticmethod
    def _plot_transductive_vs_inductive(
        tmetrics: Dict[str, Any],
        imetrics: Dict[str, Any],
        plot_dir: Path,
        save_with_caption,
        plt,
    ) -> None:
        """Grouped bar chart: 7 metrics, paired transductive vs inductive."""

        def safe_get(d: Dict[str, Any], path: List[str], default: float = float("nan")) -> float:
            cur = d
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    return default
                cur = cur[key]
            try:
                return float(cur)
            except (TypeError, ValueError):
                return default

        metric_specs: List[Tuple[str, List[str], List[str]]] = [
            ("R² GRM-ridge",        ["regression", "grm_ridge", "r2"],                                ["regression", "grm_ridge", "r2"]),
            ("R² raw-RF",           ["regression", "raw_random_forest", "r2"],                        ["regression", "raw_random_forest", "r2"]),
            ("R² naive",            ["regression", "naive_current_score", "r2"],                      ["regression", "naive_current_score", "r2"]),
            ("AUC GRM-logistic",    ["classification", "grm_logistic", "roc_auc"],                    ["classification", "grm_logistic", "roc_auc"]),
            ("AUC raw-RF",          ["classification", "raw_random_forest", "roc_auc"],               ["classification", "raw_random_forest", "roc_auc"]),
            ("AUC naive",           ["classification", "naive_current_score", "roc_auc"],             ["classification", "naive_current_score", "roc_auc"]),
            ("Latent recovery",     ["latent_recovery", "mean_abs_aligned_correlation"],              ["latent_recovery", "out_of_sample_test", "mean_abs_aligned_correlation"]),
        ]

        labels = [m[0] for m in metric_specs]
        t_vals = np.array([safe_get(tmetrics, m[1]) for m in metric_specs], dtype=float)
        i_vals = np.array([safe_get(imetrics, m[2]) for m in metric_specs], dtype=float)

        x = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width / 2, t_vals, width, label="transductive", color="#4C78A8")
        ax.bar(x + width / 2, i_vals, width, label="inductive", color="#F58518")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        for xi, t, i in zip(x, t_vals, i_vals):
            if np.isfinite(t) and np.isfinite(i):
                delta = i - t
                top = max(t, i, 0.0)
                ax.text(xi, top + 0.02, f"Δ={delta:+.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Metric value")
        ax.set_title("Transductive vs strict inductive evaluation (same dataset, seed-controlled)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        save_with_caption(fig, plot_dir / "transductive_vs_inductive_metrics.png", dpi=160)

    @staticmethod
    def _plot_flare_calibration(
        pred_df: pd.DataFrame, plot_dir: Path, save_with_caption, plt, n_bins: int = 10
    ) -> None:
        """Two reliability diagrams: raw vs temperature-calibrated flare probability."""

        y_true = pred_df["flare_next_day"].astype(int).to_numpy()
        raw = pred_df["pred_grm_flare_prob"].astype(float).to_numpy()
        cal = pred_df["pred_grm_flare_prob_calibrated"].astype(float).to_numpy()
        mask = np.isfinite(raw) & np.isfinite(cal) & np.isfinite(y_true)
        y_true = y_true[mask]
        raw = np.clip(raw[mask], 0.0, 1.0)
        cal = np.clip(cal[mask], 0.0, 1.0)
        if y_true.size < 20:
            raise RuntimeError("not enough finite predictions for reliability diagram")

        def _reliability(probs: np.ndarray) -> Tuple[List[float], List[float], List[int], float]:
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            mean_p, emp, ns = [], [], []
            ece = 0.0
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
                if not m.any():
                    continue
                mp = float(probs[m].mean())
                ep = float(y_true[m].mean())
                n = int(m.sum())
                mean_p.append(mp); emp.append(ep); ns.append(n)
                ece += (n / len(probs)) * abs(ep - mp)
            return mean_p, emp, ns, float(ece)

        raw_mp, raw_ep, raw_n, raw_ece = _reliability(raw)
        cal_mp, cal_ep, cal_n, cal_ece = _reliability(cal)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, mp, ep, ns, ece, title in [
            (axes[0], raw_mp, raw_ep, raw_n, raw_ece, "Raw flare prob"),
            (axes[1], cal_mp, cal_ep, cal_n, cal_ece, "Temperature-calibrated"),
        ]:
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
            ax.plot(mp, ep, marker="o", linewidth=1.2, color="#4C78A8")
            for mpx, epx, nx in zip(mp, ep, ns):
                ax.annotate(str(nx), (mpx, epx), fontsize=7, alpha=0.7, ha="center", va="bottom")
            ax.set_xlabel("Mean predicted prob")
            ax.set_ylabel("Empirical accuracy")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{title}  (ECE={ece:.3f})")
            ax.grid(True, linestyle=":", alpha=0.4)
        fig.suptitle("Flare-next-day reliability on inductive test set")
        fig.tight_layout()
        save_with_caption(fig, plot_dir / "flare_calibration_raw_vs_temperature.png", dpi=160)

    @staticmethod
    def _plot_per_subject_performance(
        pred_df: pd.DataFrame, plot_dir: Path, save_with_caption, plt, min_visits: int = 5
    ) -> None:
        """Box + strip plot of per-subject next-day R² and per-subject flare AUC."""

        from sklearn.metrics import r2_score, roc_auc_score

        per_subject_r2: List[float] = []
        per_subject_auc: List[float] = []
        for sid, group in pred_df.groupby("subject_id"):
            if len(group) < min_visits:
                continue
            y_reg = group["next_day_score"].to_numpy(float)
            p_reg = group.get("pred_grm_next_score")
            if p_reg is not None and np.isfinite(p_reg).all() and np.isfinite(y_reg).all():
                try:
                    per_subject_r2.append(float(r2_score(y_reg, p_reg.to_numpy(float))))
                except ValueError:
                    pass
            y_cls = group["flare_next_day"].astype(int).to_numpy()
            p_cls = group.get("pred_grm_flare_prob")
            if p_cls is not None and len(np.unique(y_cls)) > 1 and np.isfinite(p_cls).all():
                try:
                    per_subject_auc.append(float(roc_auc_score(y_cls, p_cls.to_numpy(float))))
                except ValueError:
                    pass

        if not per_subject_r2 and not per_subject_auc:
            raise RuntimeError("no subjects with sufficient visits for per-subject metrics")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, vals, title, ylim in [
            (axes[0], per_subject_r2, f"Per-subject R² next-day  (n={len(per_subject_r2)})", None),
            (axes[1], per_subject_auc, f"Per-subject flare AUC  (n={len(per_subject_auc)})", (0.0, 1.0)),
        ]:
            if not vals:
                ax.text(0.5, 0.5, "no eligible subjects", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            ax.boxplot(vals, vert=True, widths=0.5, showmeans=True, meanline=True)
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(1 + jitter, vals, s=12, alpha=0.5, color="#4C78A8")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([1])
            ax.set_xticklabels([""])
            if ylim is not None:
                ax.set_ylim(*ylim)
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
            ax.grid(axis="y", linestyle=":", alpha=0.4)
        fig.suptitle("Per-subject performance heterogeneity (inductive test set, min_visits={})".format(min_visits))
        fig.tight_layout()
        save_with_caption(fig, plot_dir / "per_subject_performance.png", dpi=160)


def _parse_cli() -> GRMTrainConfig:
    """Parse CLI args into a GRMTrainConfig. Defaults match the dataclass."""

    import argparse

    defaults = GRMTrainConfig()
    parser = argparse.ArgumentParser(description="Train the static GRM-TCM pipeline.")
    parser.add_argument("--input-dir", default=defaults.input_dir)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument("--graph-mode", default=defaults.graph_mode,
                        choices=["feature_only", "feature_only_diffusion", "temporal_only", "feature_temporal",
                                 "feature_temporal_treatment", "feature_temporal_treatment_subject",
                                 "random_graph"])
    parser.add_argument("--n-modes", type=int, default=defaults.n_modes)
    parser.add_argument("--rho", type=float, default=defaults.rho)
    parser.add_argument("--diffusion-alpha", type=float, default=defaults.diffusion_alpha)
    parser.add_argument("--test-size", type=float, default=defaults.test_size)
    parser.add_argument("--inductive", action="store_true",
                        help="Strict inductive eval: split subjects first, fit on train only, "
                             "project test subjects via --projection.")
    parser.add_argument("--projection", choices=["surrogate", "nystrom"], default=defaults.projection,
                        help="Inductive-only: how to project test-subject visits.")
    parser.add_argument("--n-neighbors-inductive", type=int, default=defaults.n_neighbors_inductive)
    args = parser.parse_args()
    return GRMTrainConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        graph_mode=args.graph_mode,
        n_modes=args.n_modes,
        rho=args.rho,
        diffusion_alpha=args.diffusion_alpha,
        test_size=args.test_size,
        inductive=args.inductive,
        projection=args.projection,
        n_neighbors_inductive=args.n_neighbors_inductive,
    )


if __name__ == "__main__":
    cfg = _parse_cli()
    trainer = GRMTCMTrainer(cfg)
    metrics = trainer.run()
    print(f"\nGRM-TCM training complete ({'inductive' if cfg.inductive else 'transductive'}).")
    print(f"Results written to: {Path(cfg.output_dir).resolve()}")
    print(json.dumps(metrics, indent=2))
