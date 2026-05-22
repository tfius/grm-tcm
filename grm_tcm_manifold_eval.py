from __future__ import annotations

"""Evaluate GRM graph modes against true manifold Laplace-Beltrami modes."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (x - mu) / sd


def _corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _standardize(a)
    b = _standardize(b)
    return (a.T @ b) / max(a.shape[0] - 1, 1)


def evaluate_manifold_alignment(data_dir: Path, results_dir: Path, output_dir: Path) -> Dict[str, object]:
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = results_dir / "grm_visit_embeddings.csv"
    lbo_path = data_dir / "true_lbo_modes.csv"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing {embeddings_path}; run grm_tcm_train.py on the manifold data first.")
    if not lbo_path.exists():
        raise FileNotFoundError(f"Missing {lbo_path}; run grm_tcm_manifold_generator.py first.")

    emb = pd.read_csv(embeddings_path)
    lbo = pd.read_csv(lbo_path)
    mode_cols = [c for c in emb.columns if c.startswith("grm_mode_")]
    lbo_cols = [c for c in lbo.columns if c.startswith("true_lbo_mode_")]
    if not mode_cols or not lbo_cols:
        raise ValueError("Missing GRM or true LBO mode columns.")

    joined = emb[["subject_id", "day", *mode_cols]].merge(
        lbo[["subject_id", "day", *lbo_cols]],
        on=["subject_id", "day"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) < 10:
        raise ValueError(f"Too few joined visits for alignment: {len(joined)}")

    X = joined[mode_cols].to_numpy(float)
    # Drop the constant LBO mode when present; graph embeddings also drop the null mode.
    lbo_use_cols = lbo_cols[1:] if len(lbo_cols) > 1 else lbo_cols
    Y = joined[lbo_use_cols].to_numpy(float)

    C = _corr_matrix(X, Y)
    rows: List[Dict[str, object]] = []
    for i, col in enumerate(mode_cols):
        j = int(np.nanargmax(np.abs(C[i])))
        rows.append({
            "grm_mode": col,
            "best_lbo_mode": lbo_use_cols[j],
            "abs_corr": float(abs(C[i, j])),
            "corr": float(C[i, j]),
        })
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(output_dir / "grm_lbo_mode_alignment.csv", index=False)

    q = min(X.shape[1], Y.shape[1])
    Xq = _standardize(X[:, :q])
    Yq = _standardize(Y[:, :q])
    R, _ = orthogonal_procrustes(Xq, Yq)
    aligned = Xq @ R
    ss_res = float(np.sum((Yq - aligned) ** 2))
    ss_tot = float(np.sum((Yq - np.mean(Yq, axis=0, keepdims=True)) ** 2))
    procrustes_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    metrics: Dict[str, object] = {
        "n_joined_visits": int(len(joined)),
        "n_grm_modes": int(len(mode_cols)),
        "n_lbo_modes_used": int(len(lbo_use_cols)),
        "mean_best_abs_corr": float(corr_df["abs_corr"].mean()),
        "median_best_abs_corr": float(corr_df["abs_corr"].median()),
        "max_best_abs_corr": float(corr_df["abs_corr"].max()),
        "procrustes_r2_first_q_modes": float(procrustes_r2),
        "q_modes": int(q),
        "note": (
            "Compares saved static GRM embeddings against analytical torus LBO modes. "
            "The constant LBO mode is dropped before scoring because GRM embeddings skip the null mode."
        ),
    }
    with open(output_dir / "manifold_lbo_alignment_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GRM mode alignment to true manifold LBO modes.")
    parser.add_argument("--data-dir", default="synthetic_grm_tcm_manifold")
    parser.add_argument("--results-dir", default="grm_tcm_results_manifold")
    parser.add_argument("--output-dir", default="grm_tcm_manifold_eval")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate_manifold_alignment(Path(args.data_dir), Path(args.results_dir), Path(args.output_dir))
    print(json.dumps(metrics, indent=2))
