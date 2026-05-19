from __future__ import annotations

"""
GRM-TCM Tier 1.5 — post-hoc recalibration of F and GRM transition probabilities.

The Tier-1 autotune showed that pure blending can't fix F's catastrophic
high-confidence miscalibration: the blend shifts the cliff down without
flattening it. The right tool for that failure mode is recalibration — the
top-1 argmax stays the same, but the probability magnitudes are remapped.

This script reads the cached per-visit predictions from `grm_tcm_autotune/`,
applies three subject-CV-honest recalibration methods to F and GRM, and asks:

  1. Does temperature scaling beat raw F on log-loss / ECE?
  2. Does isotonic regression (non-parametric) beat temperature scaling?
  3. After recalibrating F, does any blend with GRM still help?
  4. Does recalibrating GRM change anything? (Expected: little — GRM is
     already calibrated; calibration is a remap that can't add information.)

Run:
  python grm_tcm_autotune.py            # produces the cached predictions
  python grm_tcm_autotune_calibrate.py  # consumes them, runs Tier 1.5

Outputs (under grm_tcm_autotune/calibration/):
  calibration_metrics.csv               -- per-model metrics with CIs
  calibration_certificate.json          -- structured PASS/FAIL verdicts
  plots/before_after_reliability.png    -- F + GRM reliability before/after
  plots/log_loss_calibration_bar.png    -- log-loss bar chart per recalibration

Scientific framing: same as Tier 1. A "winning" recalibration here is evidence
about the F model on this synthetic benchmark, not a clinical claim.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar, minimize
from scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from grm_tcm_dynamic_eval import (
    _multi_brier,
    _topk_accuracy,
    cluster_bootstrap_paired,
    expected_calibration_error,
    reliability_table,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CalibrateConfig:
    """Knobs for the Tier 1.5 run."""

    autotune_dir: Path = Path("grm_tcm_autotune")
    output_dir: Path = Path("grm_tcm_autotune/calibration")
    seed: int = 42
    cv_splits: int = 5
    bootstrap_n: int = 200


def parse_args() -> CalibrateConfig:
    """Parse CLI."""
    p = argparse.ArgumentParser(description="Tier 1.5 recalibration of F and GRM.")
    p.add_argument("--autotune-dir", type=Path, default=Path("grm_tcm_autotune"))
    p.add_argument("--output-dir", type=Path, default=Path("grm_tcm_autotune/calibration"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--bootstrap-n", type=int, default=200)
    args = p.parse_args()
    return CalibrateConfig(
        autotune_dir=args.autotune_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        cv_splits=args.cv_splits,
        bootstrap_n=args.bootstrap_n,
    )


# ---------------------------------------------------------------------------
# Calibration methods
# ---------------------------------------------------------------------------


def _to_logits(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map probabilities to logits (up to additive constant per row)."""
    return np.log(np.clip(probs, eps, 1.0))


def _softmax_from_logits(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Single scalar T > 0 minimizing NLL of softmax(logits / T)."""
    logits = _to_logits(probs)

    def nll(t: float) -> float:
        scaled = logits / max(float(t), 1e-3)
        log_p = scaled - logsumexp(scaled, axis=1, keepdims=True)
        return float(-np.mean(log_p[np.arange(len(y)), y]))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Recalibrate probs by dividing logits by T and re-softmax."""
    logits = _to_logits(probs)
    return _softmax_from_logits(logits / max(float(T), 1e-3))


def fit_vector_scaling(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-class temperatures (vector scaling): one T_c per class, minimizing NLL.

    Returns a length-K vector of positive temperatures.
    """
    n_classes = probs.shape[1]
    logits = _to_logits(probs)

    def nll(t_vec: np.ndarray) -> float:
        t_safe = np.maximum(t_vec, 1e-3)
        scaled = logits / t_safe.reshape(1, -1)
        log_p = scaled - logsumexp(scaled, axis=1, keepdims=True)
        return float(-np.mean(log_p[np.arange(len(y)), y]))

    x0 = np.ones(n_classes, dtype=float)
    res = minimize(nll, x0, method="L-BFGS-B", bounds=[(0.05, 10.0)] * n_classes)
    return np.maximum(res.x, 1e-3)


def apply_vector_scaling(probs: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    """Recalibrate per-class with per-class temperature."""
    logits = _to_logits(probs)
    return _softmax_from_logits(logits / np.maximum(t_vec, 1e-3).reshape(1, -1))


def fit_apply_isotonic(
    probs_train: np.ndarray, y_train: np.ndarray, probs_apply: np.ndarray
) -> np.ndarray:
    """One-vs-rest isotonic regression per class, then renormalize.

    Non-parametric — can fix non-monotonic miscalibration that temperature scaling
    can't. May overfit on small classes; clip out-of-bounds.
    """
    n_classes = probs_train.shape[1]
    out = np.zeros_like(probs_apply)
    for c in range(n_classes):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        iso.fit(probs_train[:, c], (y_train == c).astype(float))
        out[:, c] = iso.transform(probs_apply[:, c])
    out = np.clip(out, 1e-12, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def _cv_apply(
    probs: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    cv_splits: int,
    fit_apply: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Generic subject-grouped CV wrapper: fit on train fold, apply to test fold.

    `fit_apply(probs_train, y_train, probs_test) -> probs_test_recalibrated`.
    """
    out = np.zeros_like(probs)
    gkf = GroupKFold(n_splits=cv_splits)
    for tr, te in gkf.split(probs, y, groups=subject_ids):
        out[te] = fit_apply(probs[tr], y[tr], probs[te])
    return out


def calibrate_temperature_cv(probs, y, subject_ids, cv_splits):
    """Subject-CV temperature scaling: fit T on train, apply to test."""

    def fit_apply(p_tr, y_tr, p_te):
        T = fit_temperature(p_tr, y_tr)
        return apply_temperature(p_te, T)

    return _cv_apply(probs, y, subject_ids, cv_splits, fit_apply)


def calibrate_vector_scaling_cv(probs, y, subject_ids, cv_splits):
    """Subject-CV per-class temperature scaling."""

    def fit_apply(p_tr, y_tr, p_te):
        t_vec = fit_vector_scaling(p_tr, y_tr)
        return apply_vector_scaling(p_te, t_vec)

    return _cv_apply(probs, y, subject_ids, cv_splits, fit_apply)


def calibrate_isotonic_cv(probs, y, subject_ids, cv_splits):
    """Subject-CV per-class isotonic regression."""
    return _cv_apply(probs, y, subject_ids, cv_splits, fit_apply_isotonic)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_probs(
    probs: np.ndarray, y: np.ndarray, n_states: int
) -> Dict[str, float]:
    """Log-loss, Brier, top-1, top-2, ECE on full-class probability predictions."""
    safe = np.clip(probs, 1e-12, 1.0)
    safe = safe / safe.sum(axis=1, keepdims=True)
    rows = np.arange(len(y))
    return {
        "log_loss": float(-np.mean(np.log(safe[rows, y]))),
        "brier": _multi_brier(y, safe, n_states),
        "top1_acc": float(np.mean(safe.argmax(axis=1) == y)),
        "top2_acc": _topk_accuracy(y, safe, k=2),
        "ece": expected_calibration_error(y, safe, n_bins=10),
    }


def bootstrap_log_loss_ece(
    probs: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, n_boot: int, seed: int, label: str,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Cluster bootstrap log-loss and ECE simultaneously over subjects."""
    safe = np.clip(probs, 1e-12, 1.0)
    rows_all = np.arange(len(y))

    def paired(idx: np.ndarray) -> Tuple[float, float]:
        if idx.size == 0:
            return (float("nan"), float("nan"))
        rows = rows_all[: len(idx)]  # used only as an index template
        ll = float(-np.mean(np.log(safe[idx, y[idx]])))
        ece = expected_calibration_error(y[idx], safe[idx], n_bins=10)
        return (ll, ece)

    res = cluster_bootstrap_paired(subject_ids, paired, n_boot=n_boot, seed=seed, label=label)
    return res[0], res[1]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_predictions(autotune_dir: Path) -> Dict[str, Any]:
    """Read the cached per-visit predictions parquet produced by Tier 1."""
    path = autotune_dir / "per_visit_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python grm_tcm_autotune.py` first to produce it."
        )
    df = pd.read_parquet(path)
    P_F_cols = sorted([c for c in df.columns if c.startswith("P_F_")], key=lambda s: int(s.split("_")[-1]))
    P_GRM_cols = sorted([c for c in df.columns if c.startswith("P_GRM_")], key=lambda s: int(s.split("_")[-1]))
    return {
        "y": df["y_true"].to_numpy(int),
        "subject_id": df["subject_id"].to_numpy(int),
        "density": df["density"].to_numpy(float),
        "current_state": df["current_state"].to_numpy(int),
        "end_day": df["end_day"].to_numpy(int),
        "probs_F": df[P_F_cols].to_numpy(float),
        "probs_GRM": df[P_GRM_cols].to_numpy(float),
        "n_states": len(P_F_cols),
        "df": df,
    }


def run_models(data: Dict[str, Any], cfg: CalibrateConfig) -> Dict[str, np.ndarray]:
    """Apply each recalibration method to F and GRM, return a dict of probability arrays."""
    y = data["y"]
    subj = data["subject_id"]
    out: Dict[str, np.ndarray] = {
        "F_raw": data["probs_F"],
        "F_temp": calibrate_temperature_cv(data["probs_F"], y, subj, cfg.cv_splits),
        "F_vector": calibrate_vector_scaling_cv(data["probs_F"], y, subj, cfg.cv_splits),
        "F_isotonic": calibrate_isotonic_cv(data["probs_F"], y, subj, cfg.cv_splits),
        "GRM_raw": data["probs_GRM"],
        "GRM_temp": calibrate_temperature_cv(data["probs_GRM"], y, subj, cfg.cv_splits),
        "GRM_isotonic": calibrate_isotonic_cv(data["probs_GRM"], y, subj, cfg.cv_splits),
    }
    return out


def post_blend_calibrated(models: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """A handful of blends using the *calibrated* F + GRM, mirroring the Tier-1 winners."""
    out: Dict[str, np.ndarray] = {}
    for f_key in ["F_temp", "F_isotonic"]:
        for grm_key in ["GRM_raw", "GRM_isotonic"]:
            for beta in [0.5, 0.7, 0.9]:
                name = f"blend_linear_{f_key}_{grm_key}_beta_{beta:.2f}"
                blend = beta * models[f_key] + (1.0 - beta) * models[grm_key]
                out[name] = blend / blend.sum(axis=1, keepdims=True)
    return out


def evaluate_all(
    models: Dict[str, np.ndarray],
    blends: Dict[str, np.ndarray],
    data: Dict[str, Any],
    cfg: CalibrateConfig,
) -> pd.DataFrame:
    """Compute metrics + bootstrap CIs for every model and blend."""
    rows: List[Dict[str, Any]] = []
    all_models = {**models, **blends}
    for name, probs in all_models.items():
        metrics = evaluate_probs(probs, data["y"], data["n_states"])
        (ll_pt, ll_lo, ll_hi), (ece_pt, ece_lo, ece_hi) = bootstrap_log_loss_ece(
            probs, data["y"], data["subject_id"], cfg.bootstrap_n, cfg.seed, label=name,
        )
        rows.append({
            "model": name,
            **metrics,
            "log_loss_ci_low": ll_lo,
            "log_loss_ci_high": ll_hi,
            "ece_ci_low": ece_lo,
            "ece_ci_high": ece_hi,
        })
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_before_after_reliability(
    models: Dict[str, np.ndarray], data: Dict[str, Any], plot_dir: Path
) -> None:
    """Two panels: F before/after, GRM before/after."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, prefix, title in [
        (axes[0], "F", "F: raw vs recalibrated"),
        (axes[1], "GRM", "GRM: raw vs recalibrated"),
    ]:
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        for name, probs in models.items():
            if not name.startswith(prefix + "_"):
                continue
            rel = reliability_table(data["y"], probs)
            rel = rel[rel["n"] > 0]
            if rel.empty:
                continue
            ax.plot(rel["mean_confidence"], rel["empirical_accuracy"], marker="o", label=name)
        ax.set_xlabel("Mean predicted top-1 probability")
        ax.set_ylabel("Empirical top-1 accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Calibration before / after recalibration (subject-CV)")
    _save(fig, plot_dir / "before_after_reliability.png")


def plot_log_loss_calibration_bar(leaderboard: pd.DataFrame, plot_dir: Path) -> None:
    """Bar chart of log-loss per model with bootstrap 95% CI whiskers."""
    fig, ax = plt.subplots(figsize=(11, 5))
    df = leaderboard.copy()
    is_F = df["model"].str.startswith("F_")
    is_GRM = df["model"].str.startswith("GRM_")
    df = df[is_F | is_GRM].sort_values("log_loss")
    y = df["log_loss"].to_numpy(float)
    err_lo = y - df["log_loss_ci_low"].to_numpy(float)
    err_hi = df["log_loss_ci_high"].to_numpy(float) - y
    ax.bar(df["model"], y, yerr=[err_lo, err_hi], capsize=3)
    ax.set_ylabel("Log-loss (lower is better)")
    ax.set_title("Log-loss before / after recalibration (subject-CV, bootstrap 95% CI)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, plot_dir / "log_loss_calibration_bar.png")


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------


def _verdict(magnitude: float, ci_low: float, ci_high: float, *, positive_is_pass: bool, description: str) -> Dict[str, Any]:
    """Wrap a metric + CI into a structured verdict."""
    if positive_is_pass:
        passes = True if (np.isfinite(ci_low) and ci_low > 0) else False if (np.isfinite(ci_high) and ci_high < 0) else None
    else:
        passes = True if (np.isfinite(ci_high) and ci_high < 0) else False if (np.isfinite(ci_low) and ci_low > 0) else None
    return {
        "magnitude": float(magnitude),
        "ci_low": float(ci_low) if np.isfinite(ci_low) else None,
        "ci_high": float(ci_high) if np.isfinite(ci_high) else None,
        "passes": passes,
        "description": description,
    }


def write_certificate(leaderboard: pd.DataFrame, models: Dict[str, np.ndarray], data: Dict[str, Any], cfg: CalibrateConfig) -> Dict[str, Any]:
    """Structured verdicts: did each recalibration help F? Did blends still pay after recalibration?"""
    lb = leaderboard.set_index("model")
    cert: Dict[str, Any] = {"verdicts": {}, "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}}

    def diff_lift(base: str, contender: str, metric: str, positive_is_pass_when_lower: bool) -> Tuple[float, float, float]:
        """Difference of means and CIs: positive = contender LOWER than base."""
        if base not in lb.index or contender not in lb.index:
            return float("nan"), float("nan"), float("nan")
        base_pt = float(lb.loc[base, metric])
        cont_pt = float(lb.loc[contender, metric])
        base_lo = float(lb.loc[base, f"{metric}_ci_low"])
        base_hi = float(lb.loc[base, f"{metric}_ci_high"])
        cont_lo = float(lb.loc[contender, f"{metric}_ci_low"])
        cont_hi = float(lb.loc[contender, f"{metric}_ci_high"])
        # Conservative CI on (base - contender)
        return base_pt - cont_pt, base_lo - cont_hi, base_hi - cont_lo

    for recal in ["F_temp", "F_vector", "F_isotonic"]:
        mag, lo, hi = diff_lift("F_raw", recal, "log_loss", positive_is_pass_when_lower=True)
        cert["verdicts"][f"{recal}_beats_F_raw_log_loss"] = _verdict(
            mag, lo, hi, positive_is_pass=True,
            description=f"{recal} log-loss vs raw F (positive = recalibration reduced log-loss).",
        )
        mag, lo, hi = diff_lift("F_raw", recal, "ece", positive_is_pass_when_lower=True)
        cert["verdicts"][f"{recal}_beats_F_raw_ece"] = _verdict(
            mag, lo, hi, positive_is_pass=True,
            description=f"{recal} ECE vs raw F (positive = recalibration reduced ECE).",
        )
    for recal in ["GRM_temp", "GRM_isotonic"]:
        mag, lo, hi = diff_lift("GRM_raw", recal, "log_loss", positive_is_pass_when_lower=True)
        cert["verdicts"][f"{recal}_beats_GRM_raw_log_loss"] = _verdict(
            mag, lo, hi, positive_is_pass=True,
            description=f"{recal} log-loss vs raw GRM (positive = recalibration reduced log-loss).",
        )

    # Did the best blend still beat the best recalibrated F?
    blend_rows = leaderboard[leaderboard["model"].str.startswith("blend_")]
    if not blend_rows.empty:
        best_blend = blend_rows.iloc[0]
        best_F_recal = lb.loc[[m for m in ["F_temp", "F_vector", "F_isotonic"] if m in lb.index]].sort_values("log_loss").iloc[0]
        mag = float(best_F_recal["log_loss"] - best_blend["log_loss"])
        # CI conservative
        lo = float(best_F_recal["log_loss_ci_low"] - best_blend["log_loss_ci_high"])
        hi = float(best_F_recal["log_loss_ci_high"] - best_blend["log_loss_ci_low"])
        cert["verdicts"]["blend_still_beats_recalibrated_F"] = _verdict(
            mag, lo, hi, positive_is_pass=True,
            description=f"Best blend ({best_blend['model']}) log-loss vs best recalibrated F ({best_F_recal.name}).",
        )
    cert["framing"] = (
        "Tier 1.5 recalibration on a synthetic benchmark. Verdicts describe whether post-hoc remap of "
        "transition probabilities helps log-loss/ECE on this dataset. Not a clinical claim."
    )
    return cert


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------


def print_top(leaderboard: pd.DataFrame, n: int = 12) -> None:
    """Print headline log-loss/ECE per model in rank order."""
    head = leaderboard.head(n)
    print()
    print("=" * 86)
    print(f"Top {n} models by log-loss (Tier 1.5 recalibration)")
    print("=" * 86)
    print(f"{'rank':>4} {'model':36s} {'log_loss':>9} {'95% CI':>17} {'ECE':>7} {'top1':>6}")
    for r, (_, row) in enumerate(head.iterrows(), start=1):
        ci = f"[{row['log_loss_ci_low']:.3f},{row['log_loss_ci_high']:.3f}]"
        print(f"{r:>4} {row['model']:36s} {row['log_loss']:>9.4f} {ci:>17} {row['ece']:>7.4f} {row['top1_acc']:>6.3f}")


def print_verdicts(cert: Dict[str, Any]) -> None:
    """Print structured pass/fail verdicts at the end."""
    print()
    print("=" * 86)
    print("Recalibration verdicts")
    print("=" * 86)
    for key, v in cert.get("verdicts", {}).items():
        if not isinstance(v, dict):
            continue
        verdict_str = "PASS" if v.get("passes") is True else "FAIL" if v.get("passes") is False else "MARGINAL"
        lo, hi = v.get("ci_low"), v.get("ci_high")
        ci = f"({lo:+.4f}, {hi:+.4f})" if lo is not None and hi is not None else ""
        print(f"  [{verdict_str:8s}] {key:48s} Δ={v.get('magnitude'):+.4f} {ci}")


def main() -> None:
    """Run Tier 1.5 end-to-end."""
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading cached predictions from {cfg.autotune_dir}/per_visit_predictions.parquet")
    data = load_predictions(cfg.autotune_dir)
    print(f"  n_visits={len(data['y'])}  n_states={data['n_states']}  n_subjects={len(np.unique(data['subject_id']))}")

    print("[1/3] Running recalibration methods (temperature, vector, isotonic)")
    models = run_models(data, cfg)

    print("[2/3] Building post-recalibration blends")
    blends = post_blend_calibrated(models)

    print(f"[3/3] Evaluating {len(models) + len(blends)} models with cluster bootstrap")
    leaderboard = evaluate_all(models, blends, data, cfg)
    leaderboard.to_csv(cfg.output_dir / "calibration_metrics.csv", index=False)

    cert = write_certificate(leaderboard, models, data, cfg)
    with open(cfg.output_dir / "calibration_certificate.json", "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2, default=str)

    plot_dir = cfg.output_dir / "plots"
    plot_before_after_reliability(models, data, plot_dir)
    plot_log_loss_calibration_bar(leaderboard, plot_dir)

    print_top(leaderboard, n=12)
    print_verdicts(cert)
    print()
    print(f"Outputs written to: {cfg.output_dir.resolve()}")
    print("Files: calibration_metrics.csv, calibration_certificate.json, plots/*.png")


if __name__ == "__main__":
    main()
