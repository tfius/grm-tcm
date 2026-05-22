from __future__ import annotations

"""Experiment matrix for separating prediction and geometry claims.

The default suite runs the manifold benchmark across static graph modes and
writes one leaderboard with both predictive metrics and LBO-recovery metrics.
This is intentionally lightweight: it orchestrates existing generator, trainer,
and manifold-eval modules rather than adding another model.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from grm_tcm_manifold_eval import evaluate_manifold_alignment
from grm_tcm_manifold_generator import ManifoldGeneratorConfig, SyntheticGRMTCMManifoldGenerator
from grm_tcm_train import GRMTCMTrainer, GRMTrainConfig


@dataclass
class ExperimentConfig:
    suite: str = "manifold"
    data_dir: Path = Path("synthetic_grm_tcm_manifold")
    output_dir: Path = Path("grm_tcm_experiments")
    graph_modes: str = "feature_only,feature_only_diffusion,feature_temporal_treatment"
    generate: bool = False
    n_subjects: int = 200
    n_days: int = 120
    n_lbo_modes: int = 24
    n_modes: int = 8
    n_neighbors: int = 30
    alpha: float = 1.0
    seed: int = 42


def _get(d: Dict, path: List[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _largest_subspace(block: Dict) -> Dict:
    subspaces = block.get("cumulative_subspace") if isinstance(block, dict) else None
    if not isinstance(subspaces, dict) or not subspaces:
        return {}
    key = max(subspaces, key=lambda k: int(str(k).split("_")[-1]))
    return subspaces.get(key, {}) or {}


def _manifold_row(name: str, metrics: Dict, geom: Dict) -> Dict[str, object]:
    saved = _get(geom, ["candidates", "saved_static_grm"], {}) or {}
    oracle = _get(geom, ["candidates", "oracle_torus_diffusion"], {}) or {}
    obs = _get(geom, ["candidates", "observation_diffusion"], {}) or {}
    saved_subspace = _largest_subspace(saved)
    oracle_subspace = _largest_subspace(oracle)
    obs_subspace = _largest_subspace(obs)
    return {
        "experiment": name,
        "graph_mode": name,
        "next_day_grm_plus_lag_r2": _get(metrics, ["regression", "grm_plus_lag_ridge", "r2"]),
        "next_day_grm_ridge_r2": _get(metrics, ["regression", "grm_ridge", "r2"]),
        "next_day_raw_rf_r2": _get(metrics, ["regression", "raw_random_forest", "r2"]),
        "next_day_persistence_r2": _get(metrics, ["regression", "persistence_yesterday_score", "r2"]),
        "flare_grm_plus_lag_auc": _get(metrics, ["classification", "grm_plus_lag_logistic", "roc_auc"]),
        "flare_raw_rf_auc": _get(metrics, ["classification", "raw_random_forest", "roc_auc"]),
        "constitution_grm_mean_r2": _get(metrics, ["constitution_recovery", "mean_r2", "grm_aggregate_ridge"]),
        "constitution_raw_mean_r2": _get(metrics, ["constitution_recovery", "mean_r2", "raw_aggregate_ridge"]),
        "saved_grm_lbo_mean_best_abs_corr": saved.get("mean_best_abs_corr"),
        "saved_grm_lbo_largest_subspace_mean_cos2": saved_subspace.get("mean_cos2"),
        "saved_grm_lbo_largest_subspace_projection_distance": saved_subspace.get("projection_distance"),
        "oracle_lbo_largest_subspace_mean_cos2": oracle_subspace.get("mean_cos2"),
        "observation_lbo_largest_subspace_mean_cos2": obs_subspace.get("mean_cos2"),
    }


def run_manifold_suite(cfg: ExperimentConfig) -> pd.DataFrame:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.generate or not (cfg.data_dir / "visits.csv").exists():
        gen_cfg = ManifoldGeneratorConfig(
            n_subjects=cfg.n_subjects,
            n_days=cfg.n_days,
            n_lbo_modes=cfg.n_lbo_modes,
            random_seed=cfg.seed,
            output_dir=str(cfg.data_dir),
        )
        SyntheticGRMTCMManifoldGenerator(gen_cfg).run()

    rows: List[Dict[str, object]] = []
    graph_modes = [m.strip() for m in cfg.graph_modes.split(",") if m.strip()]
    for graph_mode in graph_modes:
        run_dir = cfg.output_dir / f"manifold_{graph_mode}"
        print(f"[train] graph_mode={graph_mode} -> {run_dir}")
        train_cfg = GRMTrainConfig(
            input_dir=str(cfg.data_dir),
            output_dir=str(run_dir),
            graph_mode=graph_mode,
            n_modes=cfg.n_modes,
            random_seed=cfg.seed,
        )
        metrics = GRMTCMTrainer(train_cfg).run()
        geom_dir = cfg.output_dir / f"manifold_{graph_mode}_lbo"
        print(f"[geometry] graph_mode={graph_mode} -> {geom_dir}")
        geom = evaluate_manifold_alignment(
            cfg.data_dir,
            run_dir,
            geom_dir,
            n_modes=cfg.n_modes,
            n_neighbors=cfg.n_neighbors,
            alpha=cfg.alpha,
        )
        rows.append(_manifold_row(graph_mode, metrics, geom))

    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(cfg.output_dir / "manifold_graph_mode_leaderboard.csv", index=False)
    with open(cfg.output_dir / "manifest.json", "w", encoding="utf-8") as f:
        payload = asdict(cfg)
        payload["data_dir"] = str(cfg.data_dir)
        payload["output_dir"] = str(cfg.output_dir)
        json.dump(payload, f, indent=2)
    return leaderboard


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run GRM-TCM experiment matrices.")
    parser.add_argument("--suite", default="manifold", choices=["manifold"])
    parser.add_argument("--data-dir", type=Path, default=Path("synthetic_grm_tcm_manifold"))
    parser.add_argument("--output-dir", type=Path, default=Path("grm_tcm_experiments"))
    parser.add_argument("--graph-modes", default="feature_only,feature_only_diffusion,feature_temporal_treatment")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--n-subjects", type=int, default=200)
    parser.add_argument("--n-days", type=int, default=120)
    parser.add_argument("--n-lbo-modes", type=int, default=24)
    parser.add_argument("--n-modes", type=int, default=8)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return ExperimentConfig(
        suite=args.suite,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        graph_modes=args.graph_modes,
        generate=args.generate,
        n_subjects=args.n_subjects,
        n_days=args.n_days,
        n_lbo_modes=args.n_lbo_modes,
        n_modes=args.n_modes,
        n_neighbors=args.n_neighbors,
        alpha=args.alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    if config.suite == "manifold":
        df = run_manifold_suite(config)
    else:
        raise ValueError(f"Unknown suite: {config.suite}")
    print(df.to_string(index=False))
