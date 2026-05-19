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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
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


STATIC_SCHEMA_VERSION = "static-v2"


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
    temporal_edge_weight: float = 0.75
    same_subject_edge_weight: float = 0.15
    treatment_edge_weight: float = 0.20
    graph_mode: str = "feature_temporal_treatment"

    n_modes: int = 8
    rho: float = 1.0
    use_normalized_laplacian: bool = True

    test_size: float = 0.25
    target_regression: str = "next_day_score"
    target_classification: str = "flare_next_day"

    # Strict inductive evaluation: split subjects first, fit scaler/KNN/graph/
    # eigenbasis ONLY on train subjects, then project test subjects via the
    # chosen projection method ('surrogate' or 'nystrom'). Reports honest
    # held-out metrics; the persisted model is the train-only fit.
    inductive: bool = False
    projection: str = "surrogate"
    n_neighbors_inductive: int = 12


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
        self.train_degrees: Optional[np.ndarray] = None
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
            return self._run_inductive(visits, latent, events)
        return self._run_transductive(visits, latent, events)

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
        X_test_raw = test_visits[OBSERVATION_NAMES].to_numpy(float)
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
        raw_cols = OBSERVATION_NAMES + ["global_dysregulation_score"]
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

        # Naive baseline: just use the current dysregulation score.
        pred_naive_reg = test_visits["global_dysregulation_score"].to_numpy(float)
        pred_naive_reg = np.nan_to_num(pred_naive_reg, nan=float(np.nanmedian(y_reg_train)))
        naive_threshold = float(np.nanmedian(y_reg_train))
        pred_naive_cls = (pred_naive_reg >= naive_threshold).astype(int)
        prob_naive_cls = np.clip(
            pred_naive_reg / max(float(np.nanmax(y_reg_train)), 1e-9), 0, 1
        )

        # 5. Out-of-sample latent recovery: fit Procrustes on train, apply to test.
        latent_recovery = self._latent_recovery_inductive(
            train_visits, train_embeddings, test_visits, test_embeddings, latent
        )

        metrics: Dict[str, Any] = {
            "manifest": "model/manifest.json",
            "evaluation_mode": "inductive",
            "projection": self.cfg.projection,
            "n_train_subjects": int(len(train_subjects)),
            "n_test_subjects": int(len(test_subjects)),
            "n_train_visits": int(len(train_visits)),
            "n_test_visits": int(len(test_visits)),
            "regression": {
                "grm_ridge": self._reg_metrics(y_reg_test, pred_grm_reg),
                "raw_random_forest": self._reg_metrics(y_reg_test, pred_raw_reg),
                "naive_current_score": self._reg_metrics(y_reg_test, pred_naive_reg),
            },
            "classification": {
                "grm_logistic": self._cls_metrics(y_cls_test, pred_grm_cls, prob_grm_cls),
                "grm_logistic_calibrated": (
                    self._cls_metrics(y_cls_test, pred_grm_cls_cal, prob_grm_cls_cal)
                    if prob_grm_cls_cal is not None else {}
                ),
                "raw_random_forest": self._cls_metrics(y_cls_test, pred_raw_cls, prob_raw_cls),
                "naive_current_score": self._cls_metrics(y_cls_test, pred_naive_cls, prob_naive_cls),
            },
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
        return visits, latent, events

    def _prepare_visits(self, visits: pd.DataFrame) -> pd.DataFrame:
        df = visits.sort_values(["subject_id", "day"]).reset_index(drop=True).copy()
        for col in [self.cfg.target_regression, self.cfg.target_classification]:
            if col not in df.columns:
                raise ValueError(f"Missing target column: {col}")
        df = df.dropna(subset=[self.cfg.target_regression, self.cfg.target_classification]).reset_index(drop=True)
        df["visit_id"] = np.arange(len(df))
        return df

    def _make_observation_matrix(self, visits: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        missing = [c for c in OBSERVATION_NAMES if c not in visits.columns]
        if missing:
            raise ValueError(f"Missing observation columns: {missing}")
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        X = pipe.fit_transform(visits[OBSERVATION_NAMES].to_numpy(dtype=float))
        self.obs_preprocessor = pipe
        return X, OBSERVATION_NAMES

    def _build_visit_graph(self, visits: pd.DataFrame, X: np.ndarray, events: Optional[pd.DataFrame]) -> sparse.csr_matrix:
        valid_modes = {
            "feature_only",
            "temporal_only",
            "feature_temporal",
            "feature_temporal_treatment",
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
        if self.cfg.graph_mode in {"feature_only", "feature_temporal", "feature_temporal_treatment"}:
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
        if self.cfg.graph_mode in {"temporal_only", "feature_temporal", "feature_temporal_treatment"}:
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
            self.cfg.graph_mode == "feature_temporal_treatment"
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

        W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        W.setdiag(0.0)
        W.eliminate_zeros()
        return W.maximum(W.T)

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

        k = min(self.cfg.n_modes + 1, n - 2)
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
        order = np.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        return eigenvalues[1:self.cfg.n_modes + 1], eigenvectors[:, 1:self.cfg.n_modes + 1]

    def _make_grm_embeddings(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
        weights = 1.0 / (1.0 + (self.cfg.rho ** 2) * eigenvalues)
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
        pred_naive_reg = visits.iloc[test_idx]["global_dysregulation_score"].to_numpy(float)
        pred_naive_reg = np.nan_to_num(pred_naive_reg, nan=float(np.nanmedian(y_reg[train_idx])))

        pred_grm_cls, prob_grm_cls, self.logistic_clf = self._fit_cls(X_grm, y_cls, train_idx, test_idx, "logistic")
        pred_raw_cls, prob_raw_cls, _ = self._fit_cls(X_raw, y_cls, train_idx, test_idx, "random_forest")
        naive_threshold = float(np.nanmedian(y_reg[train_idx]))
        pred_naive_cls = (pred_naive_reg >= naive_threshold).astype(int)
        prob_naive_cls = np.clip(pred_naive_reg / max(float(np.nanmax(y_reg[train_idx])), 1e-9), 0, 1)

        self.flare_temperature = self._fit_flare_temperature(X_grm, y_cls, train_idx)
        prob_grm_cls_calibrated = self._apply_flare_temperature(self.logistic_clf, X_grm, test_idx, self.flare_temperature)
        pred_grm_cls_calibrated = (prob_grm_cls_calibrated >= 0.5).astype(int) if prob_grm_cls_calibrated is not None else None

        metrics: Dict = {
            "manifest": "model/manifest.json",
            "regression": {
                "grm_ridge": self._reg_metrics(y_reg[test_idx], pred_grm_reg),
                "raw_random_forest": self._reg_metrics(y_reg[test_idx], pred_raw_reg),
                "naive_current_score": self._reg_metrics(y_reg[test_idx], pred_naive_reg),
            },
            "classification": {
                "grm_logistic": self._cls_metrics(y_cls[test_idx], pred_grm_cls, prob_grm_cls),
                "grm_logistic_calibrated": (
                    self._cls_metrics(y_cls[test_idx], pred_grm_cls_calibrated, prob_grm_cls_calibrated)
                    if prob_grm_cls_calibrated is not None else {}
                ),
                "raw_random_forest": self._cls_metrics(y_cls[test_idx], pred_raw_cls, prob_raw_cls),
                "naive_current_score": self._cls_metrics(y_cls[test_idx], pred_naive_cls, prob_naive_cls),
            },
            "flare_temperature": float(self.flare_temperature) if self.flare_temperature is not None else None,
            "latent_recovery": self._latent_recovery_capture(visits, embeddings, latent) if latent is not None else {},
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
        return metrics, pred_df

    @staticmethod
    def _raw_baseline_matrix(visits: pd.DataFrame) -> np.ndarray:
        cols = OBSERVATION_NAMES + ["global_dysregulation_score"]
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
            extra=manifest_extra,
        )


def _parse_cli() -> GRMTrainConfig:
    """Parse CLI args into a GRMTrainConfig. Defaults match the dataclass."""

    import argparse

    defaults = GRMTrainConfig()
    parser = argparse.ArgumentParser(description="Train the static GRM-TCM pipeline.")
    parser.add_argument("--input-dir", default=defaults.input_dir)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument("--graph-mode", default=defaults.graph_mode,
                        choices=["feature_only", "temporal_only", "feature_temporal",
                                 "feature_temporal_treatment", "random_graph"])
    parser.add_argument("--n-modes", type=int, default=defaults.n_modes)
    parser.add_argument("--rho", type=float, default=defaults.rho)
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
