from __future__ import annotations

"""
Experiment runner for GRM-TCM synthetic benchmark stress tests.

Runs the generator, trainer, and diagnostics across random seeds, difficulty
settings, and graph/label ablations. Results are synthetic benchmark evidence
only; they do not prove TCM, Qi, or a biological mechanism.

Run:
  python grm_tcm_experiments.py
"""

import argparse
import json
import shutil
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from grm_tcm_diagnostics import DiagnosticsConfig, run as run_diagnostics
from grm_tcm_dynamic_grm import DynamicGRMConfig, run_dynamic
from grm_tcm_synthetic_generator import DIFFICULTY_PRESETS, GeneratorConfig, SyntheticGRMTCMGenerator
from grm_tcm_train import GRMTCMTrainer, GRMTrainConfig


ABLATIONS = {
    "feature_similarity_only": {"graph_mode": "feature_only", "permute_labels": False},
    "temporal_only": {"graph_mode": "temporal_only", "permute_labels": False},
    "feature_temporal": {"graph_mode": "feature_temporal", "permute_labels": False},
    "feature_temporal_treatment": {"graph_mode": "feature_temporal_treatment", "permute_labels": False},
    "random_graph_control": {"graph_mode": "random_graph", "permute_labels": False},
    "permuted_label_control": {"graph_mode": "feature_temporal_treatment", "permute_labels": True},
}


@dataclass
class ExperimentConfig:
    """Configuration for experiment sweeps."""

    output_dir: str = "grm_tcm_experiments"
    seeds: List[int] = field(default_factory=lambda: [42, 43])
    difficulties: List[str] = field(default_factory=lambda: ["easy", "medium", "hard", "chaotic"])
    ablations: List[str] = field(default_factory=lambda: list(ABLATIONS))
    n_subjects: int = 80
    n_days: int = 60
    clean: bool = False


def parse_csv_list(value: str) -> List[str]:
    """Parse a comma-separated CLI value."""

    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_list(value: str) -> List[int]:
    """Parse comma-separated integer seeds."""

    return [int(item) for item in parse_csv_list(value)]


def parse_args() -> ExperimentConfig:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run GRM-TCM synthetic experiment sweeps.")
    parser.add_argument("--output-dir", default="grm_tcm_experiments")
    parser.add_argument("--seeds", default="42,43", help="Comma-separated random seeds.")
    parser.add_argument("--difficulties", default="easy,medium,hard,chaotic")
    parser.add_argument("--ablations", default=",".join(ABLATIONS))
    parser.add_argument("--n-subjects", type=int, default=80)
    parser.add_argument("--n-days", type=int, default=60)
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before running.")
    args = parser.parse_args()
    return ExperimentConfig(
        output_dir=args.output_dir,
        seeds=parse_seed_list(args.seeds),
        difficulties=parse_csv_list(args.difficulties),
        ablations=parse_csv_list(args.ablations),
        n_subjects=args.n_subjects,
        n_days=args.n_days,
        clean=args.clean,
    )


def validate_config(cfg: ExperimentConfig) -> None:
    """Validate configured difficulties and ablations."""

    unknown_difficulties = sorted(set(cfg.difficulties) - set(DIFFICULTY_PRESETS))
    if unknown_difficulties:
        raise ValueError(f"Unknown difficulties: {unknown_difficulties}. Choose from {sorted(DIFFICULTY_PRESETS)}")
    unknown_ablations = sorted(set(cfg.ablations) - set(ABLATIONS))
    if unknown_ablations:
        raise ValueError(f"Unknown ablations: {unknown_ablations}. Choose from {sorted(ABLATIONS)}")


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file if present."""

    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write indented JSON."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_id(difficulty: str, seed: int, ablation: str) -> str:
    """Build a stable run id."""

    return f"{difficulty}_seed{seed}_{ablation}"


def maybe_permute_labels(data_dir: Path, seed: int) -> None:
    """Permute semantic labels as a negative-control ontology test."""

    visits_path = data_dir / "visits.csv"
    if not visits_path.exists():
        return
    rng = np.random.default_rng(seed + 10_000)
    visits = pd.read_csv(visits_path)
    for col in ["qi_like_label", "tcm_like_label", "contrarian_signature"]:
        if col in visits.columns:
            values = visits[col].to_numpy(copy=True)
            rng.shuffle(values)
            visits[col] = values
    visits.to_csv(visits_path, index=False)


def max_cluster_score(diagnostics_dir: Path, metric: str, target: str = "hidden_subtype") -> float:
    """Extract the best cluster score for a diagnostics target."""

    path = diagnostics_dir / "cluster_scores.csv"
    if not path.exists():
        return float("nan")
    scores = pd.read_csv(path)
    subset = scores[scores["target"] == target] if "target" in scores.columns else scores.iloc[0:0]
    if subset.empty or metric not in subset.columns:
        return float("nan")
    return float(subset[metric].max())


def ontology_entropy(diagnostics_dir: Path) -> float:
    """Return mean normalized entropy for TCM-label to hidden-subtype mismatch."""

    path = diagnostics_dir / "ontology_mismatch.csv"
    if not path.exists():
        return float("nan")
    mismatch = pd.read_csv(path)
    if mismatch.empty or "normalized_entropy" not in mismatch.columns:
        return float("nan")
    if "distribution" in mismatch.columns:
        mismatch = mismatch[mismatch["distribution"] == "tcm_like_label_to_hidden_subtype"]
    return float(mismatch["normalized_entropy"].dropna().mean()) if not mismatch.empty else float("nan")


def extract_metrics(
    cfg: ExperimentConfig,
    difficulty: str,
    seed: int,
    ablation: str,
    run_dir: Path,
) -> Dict[str, Any]:
    """Collect trainer and diagnostics metrics for one run."""

    metrics = read_json(run_dir / "results" / "grm_metrics.json")
    summary = read_json(run_dir / "diagnostics" / "diagnostics_summary.json")
    graph_mode = ABLATIONS[ablation]["graph_mode"]
    row: Dict[str, Any] = {
        "run_id": run_id(difficulty, seed, ablation),
        "difficulty": difficulty,
        "seed": seed,
        "ablation": ablation,
        "graph_mode": graph_mode,
        "permuted_labels": bool(ABLATIONS[ablation]["permute_labels"]),
        "n_subjects": cfg.n_subjects,
        "n_days": cfg.n_days,
        "run_dir": str(run_dir),
    }
    row["grm_next_day_r2"] = metrics.get("regression", {}).get("grm_ridge", {}).get("r2", np.nan)
    row["raw_next_day_r2"] = metrics.get("regression", {}).get("raw_random_forest", {}).get("r2", np.nan)
    row["naive_next_day_r2"] = metrics.get("regression", {}).get("naive_current_score", {}).get("r2", np.nan)
    row["flare_roc_auc"] = metrics.get("classification", {}).get("grm_logistic", {}).get("roc_auc", np.nan)
    row["latent_recovery_corr"] = metrics.get("latent_recovery", {}).get("mean_abs_aligned_correlation", np.nan)
    row["best_abs_grm_latent_correlation"] = summary.get("best_abs_grm_latent_correlation", np.nan)
    row["cluster_ari_hidden_subtype"] = max_cluster_score(run_dir / "diagnostics", "adjusted_rand_index")
    row["cluster_nmi_hidden_subtype"] = max_cluster_score(run_dir / "diagnostics", "normalized_mutual_info")
    row["ontology_mismatch_entropy"] = ontology_entropy(run_dir / "diagnostics")
    row["diagnostic_contrarian_findings"] = summary.get("n_contrarian_findings", np.nan)
    dynamic = read_json(run_dir / "dynamic" / "dynamic_grm_metrics.json")
    row["dynamic_regime_flare_auc"] = dynamic.get("regime_flare_auc", np.nan)
    row["dynamic_regime_crash_auc"] = dynamic.get("regime_crash_auc", np.nan)
    row["dynamic_self_resonance_flare_auc"] = dynamic.get("self_resonance_flare_auc", np.nan)
    row["dynamic_self_resonance_crash_auc"] = dynamic.get("self_resonance_crash_auc", np.nan)
    row["dynamic_soft_self_resonance_flare_auc"] = dynamic.get("soft_self_resonance_flare_auc", np.nan)
    row["dynamic_soft_self_resonance_crash_auc"] = dynamic.get("soft_self_resonance_crash_auc", np.nan)
    row["dynamic_transition_accuracy"] = dynamic.get("grm_transition_accuracy", np.nan)
    row["dynamic_markov_transition_accuracy"] = dynamic.get("markov_transition_accuracy", np.nan)
    row["dynamic_transition_accuracy_lift"] = dynamic.get("transition_accuracy_lift", np.nan)
    row["dynamic_transition_log_loss_lift"] = dynamic.get("transition_log_loss_lift", np.nan)
    row["dynamic_transition_brier_lift"] = dynamic.get("transition_brier_lift", np.nan)
    row["dynamic_transition_ece_lift"] = dynamic.get("transition_ece_lift", np.nan)
    row["dynamic_subject_regime_flare_auc"] = dynamic.get("subject_regime_flare_auc", np.nan)
    row["dynamic_subject_regime_crash_auc"] = dynamic.get("subject_regime_crash_auc", np.nan)
    row["dynamic_subject_self_resonance_flare_auc"] = dynamic.get("subject_self_resonance_flare_auc", np.nan)
    row["dynamic_subject_self_resonance_crash_auc"] = dynamic.get("subject_self_resonance_crash_auc", np.nan)
    row["dynamic_subject_soft_self_resonance_flare_auc"] = dynamic.get("subject_soft_self_resonance_flare_auc", np.nan)
    row["dynamic_subject_soft_self_resonance_crash_auc"] = dynamic.get("subject_soft_self_resonance_crash_auc", np.nan)
    row["dynamic_subject_soft_self_hidden_subtype_eta_squared"] = dynamic.get(
        "subject_soft_self_resonance_hidden_subtype_eta_squared", np.nan
    )
    row["dynamic_subject_transition_accuracy_lift"] = dynamic.get("subject_transition_accuracy_lift", np.nan)
    row["dynamic_subject_transition_log_loss_lift"] = dynamic.get("subject_transition_log_loss_lift", np.nan)
    row["dynamic_subject_transition_brier_lift"] = dynamic.get("subject_transition_brier_lift", np.nan)
    row["dynamic_subject_transition_ece_lift"] = dynamic.get("subject_transition_ece_lift", np.nan)
    return row


def run_one(cfg: ExperimentConfig, output_dir: Path, difficulty: str, seed: int, ablation: str) -> Dict[str, Any]:
    """Run generator, trainer, diagnostics, and metric extraction for one experiment."""

    rid = run_id(difficulty, seed, ablation)
    run_dir = output_dir / "runs" / rid
    data_dir = run_dir / "data"
    results_dir = run_dir / "results"
    diagnostics_dir = run_dir / "diagnostics"
    dynamic_dir = run_dir / "dynamic"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[run] {rid}")
    print("[run] generating synthetic data")
    gen_cfg = GeneratorConfig(
        n_subjects=cfg.n_subjects,
        n_days=cfg.n_days,
        random_seed=seed,
        output_dir=str(data_dir),
        difficulty=difficulty,
    )
    SyntheticGRMTCMGenerator(gen_cfg).run()

    if ABLATIONS[ablation]["permute_labels"]:
        print("[run] applying permuted label control")
        maybe_permute_labels(data_dir, seed)

    print("[run] training GRM")
    train_cfg = GRMTrainConfig(
        input_dir=str(data_dir),
        output_dir=str(results_dir),
        random_seed=seed,
        graph_mode=ABLATIONS[ablation]["graph_mode"],
    )
    GRMTCMTrainer(train_cfg).run()

    print("[run] running diagnostics")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_diagnostics(DiagnosticsConfig(data_dir=str(data_dir), results_dir=str(results_dir), output_dir=str(diagnostics_dir)))

    print("[run] running dynamic GRM")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_dynamic(DynamicGRMConfig(data_dir=str(data_dir), results_dir=str(results_dir), output_dir=str(dynamic_dir)))

    return extract_metrics(cfg, difficulty, seed, ablation, run_dir)


def go_no_go(results: pd.DataFrame) -> Dict[str, Any]:
    """Make a conservative synthetic-benchmark go/no-go recommendation."""

    primary = results[results["ablation"] == "feature_temporal_treatment"].copy()
    controls = results[results["ablation"].isin(["random_graph_control", "permuted_label_control"])].copy()
    if primary.empty:
        return {"recommendation": "NO-GO", "reason": "No primary feature_temporal_treatment runs were completed."}

    primary_latent = float(primary["latent_recovery_corr"].mean())
    primary_flare = float(primary["flare_roc_auc"].mean())
    primary_cluster = float(primary["cluster_nmi_hidden_subtype"].mean())
    primary_r2 = float(primary["grm_next_day_r2"].mean())
    primary_dynamic_self = float(primary["dynamic_self_resonance_flare_auc"].mean())
    primary_dynamic_soft_self = float(primary["dynamic_soft_self_resonance_flare_auc"].mean())
    primary_subject_dynamic_self = float(primary["dynamic_subject_self_resonance_flare_auc"].mean())
    primary_subject_dynamic_soft_self = float(primary["dynamic_subject_soft_self_resonance_flare_auc"].mean())
    primary_subject_regime = float(primary["dynamic_subject_regime_flare_auc"].mean())
    primary_transition_lift = float(primary["dynamic_transition_accuracy_lift"].mean())
    control_latent = float(controls["latent_recovery_corr"].mean()) if not controls.empty else float("nan")
    control_margin = primary_latent - control_latent if not np.isnan(control_latent) else float("nan")

    passes = [
        primary_latent >= 0.35,
        primary_flare >= 0.70
        or primary_dynamic_self >= 0.65
        or primary_dynamic_soft_self >= 0.65
        or primary_subject_dynamic_self >= 0.65
        or primary_subject_dynamic_soft_self >= 0.65,
        primary_cluster >= 0.10 or primary_transition_lift > 0.0,
        np.isnan(control_margin) or control_margin >= 0.03,
    ]
    recommendation = "GO" if all(passes) else "NO-GO"
    if recommendation == "GO" and primary_r2 < 0.0:
        reason = "Latent recovery and flare prediction pass, but next-day R2 is weak; proceed only as a latent-state benchmark."
    elif recommendation == "GO":
        reason = "Primary runs recover synthetic latent structure and beat negative controls on average."
    else:
        reason = "Primary runs do not consistently recover latent structure, predict flares, or separate from controls."

    return {
        "recommendation": recommendation,
        "reason": reason,
        "criteria": {
            "primary_mean_latent_recovery_corr": primary_latent,
            "primary_mean_flare_roc_auc": primary_flare,
            "primary_mean_cluster_nmi_hidden_subtype": primary_cluster,
            "primary_mean_grm_next_day_r2": primary_r2,
            "primary_mean_dynamic_self_resonance_flare_auc": primary_dynamic_self,
            "primary_mean_dynamic_soft_self_resonance_flare_auc": primary_dynamic_soft_self,
            "primary_mean_dynamic_subject_self_resonance_flare_auc": primary_subject_dynamic_self,
            "primary_mean_dynamic_subject_soft_self_resonance_flare_auc": primary_subject_dynamic_soft_self,
            "primary_mean_dynamic_subject_regime_flare_auc": primary_subject_regime,
            "primary_mean_dynamic_transition_accuracy_lift": primary_transition_lift,
            "control_mean_latent_recovery_corr": control_latent,
            "latent_recovery_control_margin": control_margin,
        },
        "guardrail": "This is a synthetic benchmark recommendation only; it does not validate TCM, Qi, or biology.",
    }


def summarize_results(cfg: ExperimentConfig, results: pd.DataFrame) -> Dict[str, Any]:
    """Create experiment summary JSON."""

    grouped = (
        results.groupby(["difficulty", "ablation"])
        .agg(
            runs=("run_id", "count"),
            mean_grm_next_day_r2=("grm_next_day_r2", "mean"),
            mean_raw_next_day_r2=("raw_next_day_r2", "mean"),
            mean_naive_next_day_r2=("naive_next_day_r2", "mean"),
            mean_flare_roc_auc=("flare_roc_auc", "mean"),
            mean_latent_recovery_corr=("latent_recovery_corr", "mean"),
            mean_cluster_nmi_hidden_subtype=("cluster_nmi_hidden_subtype", "mean"),
            mean_ontology_mismatch_entropy=("ontology_mismatch_entropy", "mean"),
            mean_dynamic_regime_flare_auc=("dynamic_regime_flare_auc", "mean"),
            mean_dynamic_self_resonance_flare_auc=("dynamic_self_resonance_flare_auc", "mean"),
            mean_dynamic_soft_self_resonance_flare_auc=("dynamic_soft_self_resonance_flare_auc", "mean"),
            mean_dynamic_transition_accuracy_lift=("dynamic_transition_accuracy_lift", "mean"),
            mean_dynamic_subject_regime_flare_auc=("dynamic_subject_regime_flare_auc", "mean"),
            mean_dynamic_subject_self_resonance_flare_auc=("dynamic_subject_self_resonance_flare_auc", "mean"),
            mean_dynamic_subject_soft_self_resonance_flare_auc=("dynamic_subject_soft_self_resonance_flare_auc", "mean"),
            mean_dynamic_subject_transition_accuracy_lift=("dynamic_subject_transition_accuracy_lift", "mean"),
        )
        .reset_index()
    )
    return {
        "config": asdict(cfg),
        "n_runs": int(len(results)),
        "by_difficulty_ablation": grouped.to_dict(orient="records"),
        "go_no_go": go_no_go(results),
        "interpretation_guardrail": (
            "Experiment metrics are synthetic benchmark diagnostics for latent-state recovery, ontology mismatch, "
            "and ablation behavior. They do not prove TCM, Qi, or a biological mechanism."
        ),
    }


def print_recommendation(summary: Dict[str, Any], output_dir: Path) -> None:
    """Print final experiment guidance."""

    rec = summary["go_no_go"]
    print("\nExperiment sweep complete.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print("Inspect first:")
    print("  1. experiment_results.csv")
    print("  2. experiment_summary.json")
    print("  3. runs/<run_id>/diagnostics/diagnostics_summary.json")
    print(f"\nGo/no-go recommendation: {rec['recommendation']}")
    print(f"Reason: {rec['reason']}")
    print("Guardrail: synthetic benchmark only; this does not prove TCM, Qi, or biology.")


def run(cfg: ExperimentConfig) -> None:
    """Run the configured experiment sweep."""

    validate_config(cfg)
    output_dir = Path(cfg.output_dir)
    if cfg.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    total = len(cfg.difficulties) * len(cfg.seeds) * len(cfg.ablations)
    done = 0
    for difficulty in cfg.difficulties:
        for seed in cfg.seeds:
            for ablation in cfg.ablations:
                done += 1
                print(f"\n[progress] {done}/{total}")
                rows.append(run_one(cfg, output_dir, difficulty, seed, ablation))

    results = pd.DataFrame(rows)
    results_path = output_dir / "experiment_results.csv"
    results.to_csv(results_path, index=False)
    print(f"[write] {results_path}")

    summary = summarize_results(cfg, results)
    summary_path = output_dir / "experiment_summary.json"
    write_json(summary_path, summary)
    print(f"[write] {summary_path}")
    print_recommendation(summary, output_dir)


if __name__ == "__main__":
    run(parse_args())
