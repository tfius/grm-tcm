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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import orthogonal_procrustes
from scipy.sparse.linalg import eigsh
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grm_tcm_persistence import (
    canonicalize_eigvec_signs,
    save_joblib,
    write_manifest,
)


STATIC_SCHEMA_VERSION = "static-v1"


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
        self.train_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None
        self.procrustes_R: Optional[np.ndarray] = None
        self._visit_index: Optional[pd.DataFrame] = None

    def run(self) -> Dict:
        visits, latent, events = self._load_data()
        visits = self._prepare_visits(visits)
        self._visit_index = visits[["visit_id", "subject_id", "day"]].copy()
        X_obs, feature_names = self._make_observation_matrix(visits)
        W = self._build_visit_graph(visits, X_obs, events)
        eigenvalues, eigenvectors = self._spectral_decomposition(W)
        eigenvectors = canonicalize_eigvec_signs(eigenvectors)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        embeddings = self._make_grm_embeddings(eigenvalues, eigenvectors)

        embeddings_df = self._make_embeddings_df(visits, embeddings)
        metrics, predictions_df = self._evaluate(visits, embeddings, latent)
        feature_modes_df = self._feature_mode_correlations(visits, embeddings, feature_names)
        self._write_outputs(embeddings_df, feature_modes_df, metrics, predictions_df)
        self._save_model()
        return metrics

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

        metrics: Dict = {
            "manifest": "model/manifest.json",
            "regression": {
                "grm_ridge": self._reg_metrics(y_reg[test_idx], pred_grm_reg),
                "raw_random_forest": self._reg_metrics(y_reg[test_idx], pred_raw_reg),
                "naive_current_score": self._reg_metrics(y_reg[test_idx], pred_naive_reg),
            },
            "classification": {
                "grm_logistic": self._cls_metrics(y_cls[test_idx], pred_grm_cls, prob_grm_cls),
                "raw_random_forest": self._cls_metrics(y_cls[test_idx], pred_raw_cls, prob_raw_cls),
                "naive_current_score": self._cls_metrics(y_cls[test_idx], pred_naive_cls, prob_naive_cls),
            },
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

    def _write_outputs(self, embeddings_df: pd.DataFrame, feature_modes: pd.DataFrame, metrics: Dict, predictions: pd.DataFrame) -> None:
        embeddings_df.to_csv(self.output_dir / "grm_visit_embeddings.csv", index=False)
        feature_modes.to_csv(self.output_dir / "grm_feature_modes.csv", index=False)
        predictions.to_csv(self.output_dir / "grm_predictions.csv", index=False)
        with open(self.output_dir / "grm_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def _save_model(self) -> None:
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
        )


if __name__ == "__main__":
    cfg = GRMTrainConfig()
    trainer = GRMTCMTrainer(cfg)
    metrics = trainer.run()
    print("GRM-TCM training complete.")
    print(f"Results written to: {Path(cfg.output_dir).resolve()}")
    print(json.dumps(metrics, indent=2))
