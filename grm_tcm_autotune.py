from __future__ import annotations

"""
GRM-TCM Tier-1 autotuner — post-hoc blending of F (strong baseline) and GRM
predictions on the persisted transition models.

The certificate produced by grm_tcm_dynamic_eval.py already gives every claim a
scalar + CI; this script closes the loop. For each candidate blend configuration
it computes log-loss, Brier, top-1, and ECE on the subject-CV held-out
predictions, with cluster-bootstrap CIs by subject. Output: a leaderboard CSV
and a few diagnostic plots.

Tier 1 only changes the *blend* — it does NOT retrain the static or dynamic GRM
models. Tier 2 and Tier 3 (retraining GRM hyperparameters) are planned but not
implemented here.

Blend families:
  - reference:    F, GRM, Markov-A, Markov-B (Laplace), 50/50 linear blend
  - linear:       P = β·P_F + (1-β)·P_GRM,           β ∈ {0.0, 0.1, ..., 1.0}
  - log:          log P ∝ β·log P_F + (1-β)·log P_GRM, β ∈ {0.0, 0.1, ..., 1.0}
  - gate:         if max(P_F) ≤ τ → use F, else use GRM,  τ ∈ {0.0, 0.05, ..., 1.0}
  - density:      P = σ(scale·(d − shift))·P_F + (1 − σ)·P_GRM, swept over a grid

Run:
  python grm_tcm_autotune.py
  python grm_tcm_autotune.py --bootstrap-n 500 --output-dir grm_tcm_autotune

Outputs (under grm_tcm_autotune/):
  leaderboard.csv               -- all configs, primary objective ranking
  best_per_kind.json            -- best blend within each family
  pareto_front.csv              -- log-loss vs ECE Pareto subset
  manifest.json                 -- config + git sha + input hashes
  per_visit_predictions.parquet -- the cached P_F, P_GRM, y arrays (audit)
  plots/log_loss_vs_param.png   -- one curve per blend family
  plots/log_loss_vs_ece_pareto.png
  plots/reliability_top3.png    -- reliability curves for top-3 blends

Scientific framing: this is a search procedure on a synthetic benchmark. A
"winning" blend here is evidence that, on this dataset, GRM and F have
complementary failure modes — not evidence about clinical performance.
"""

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from grm_tcm_dynamic_eval import (
    DynamicEvalConfig,
    _baseline_feature_matrix,
    _empirical_markov,
    _grm_blended_from_persisted,
    _multi_brier,
    _topk_accuracy,
    build_setup,
    cluster_bootstrap_paired,
    expected_calibration_error,
    reliability_table,
)
from grm_tcm_persistence import git_commit_info


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AutotuneConfig:
    """All knobs for the Tier-1 autotune run."""

    data_dir: Path = Path("synthetic_grm_tcm")
    static_model_dir: Path = Path("grm_tcm_results/model")
    dynamic_model_dir: Path = Path("grm_tcm_dynamic/model")
    output_dir: Path = Path("grm_tcm_autotune")
    seed: int = 42
    cv_splits: int = 5
    bootstrap_n: int = 200
    density_k: int = 10
    top_k_plot: int = 3


def parse_args() -> AutotuneConfig:
    """Parse CLI flags."""
    p = argparse.ArgumentParser(description="Tier-1 autotune over post-hoc blends of F and GRM.")
    p.add_argument("--data-dir", type=Path, default=Path("synthetic_grm_tcm"))
    p.add_argument("--static-model", type=Path, default=Path("grm_tcm_results/model"))
    p.add_argument("--dynamic-model", type=Path, default=Path("grm_tcm_dynamic/model"))
    p.add_argument("--output-dir", type=Path, default=Path("grm_tcm_autotune"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--bootstrap-n", type=int, default=200)
    p.add_argument("--density-k", type=int, default=10)
    args = p.parse_args()
    return AutotuneConfig(
        data_dir=args.data_dir,
        static_model_dir=args.static_model,
        dynamic_model_dir=args.dynamic_model,
        output_dir=args.output_dir,
        seed=args.seed,
        cv_splits=args.cv_splits,
        bootstrap_n=args.bootstrap_n,
        density_k=args.density_k,
    )


# ---------------------------------------------------------------------------
# Per-visit prediction collection (subject-CV honest for F; GRM is persisted)
# ---------------------------------------------------------------------------


def collect_per_visit_predictions(cfg: AutotuneConfig) -> Dict[str, Any]:
    """Run subject-grouped CV once; return aligned per-visit P_F, P_GRM, P_A, y."""
    eval_cfg = DynamicEvalConfig(
        data_dir=cfg.data_dir,
        static_model_dir=cfg.static_model_dir,
        dynamic_model_dir=cfg.dynamic_model_dir,
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        cv_splits=cfg.cv_splits,
    )
    setup = build_setup(eval_cfg)
    visits = setup.visits
    n_states = setup.state_weights.shape[1]

    mask = visits["next_state_id"].notna() & visits["g_end_day"].ge(0)
    eval_df = visits[mask].reset_index(drop=True)
    current = eval_df["state_id"].to_numpy(int)
    nxt = eval_df["next_state_id"].to_numpy(int)
    subj = eval_df["subject_id"].to_numpy(int)
    end_days = eval_df["g_end_day"].to_numpy(int)
    n_eval = len(eval_df)
    print(f"[collect] eligible visits: {n_eval} / {len(visits)}")

    baseline_X = _baseline_feature_matrix(eval_df, n_states)
    grm_full = _grm_blended_from_persisted(setup.dynamic, current, end_days)

    probs_F = np.full((n_eval, n_states), 1.0 / n_states, dtype=float)
    probs_A = np.full((n_eval, n_states), 1.0 / n_states, dtype=float)
    probs_B = np.full((n_eval, n_states), 1.0 / n_states, dtype=float)

    gkf = GroupKFold(n_splits=cfg.cv_splits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(current, nxt, groups=subj)):
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=cfg.seed + fold_idx)
            clf.fit(baseline_X[train_idx], nxt[train_idx])
            prob_F = clf.predict_proba(baseline_X[test_idx])
            full_F = np.full((len(test_idx), n_states), 1e-6, dtype=float)
            for i, cls in enumerate(clf.classes_):
                full_F[:, int(cls)] = prob_F[:, i]
            full_F = full_F / full_F.sum(axis=1, keepdims=True)
            probs_F[test_idx] = full_F

            M_a = _empirical_markov(current[train_idx], nxt[train_idx], n_states, alpha=0.0)
            M_b = _empirical_markov(current[train_idx], nxt[train_idx], n_states, alpha=1.0)
            probs_A[test_idx] = M_a[current[test_idx]]
            probs_B[test_idx] = M_b[current[test_idx]]

    return {
        "y_true": nxt,
        "subject_id": subj,
        "current_state": current,
        "end_days": end_days,
        "probs_F": probs_F,
        "probs_GRM": grm_full,
        "probs_A_markov": probs_A,
        "probs_B_markov_laplace": probs_B,
        "n_states": n_states,
        "n_eval": n_eval,
        "baseline_features": baseline_X,
        "setup": setup,
    }


# ---------------------------------------------------------------------------
# Blend families
# ---------------------------------------------------------------------------


def blend_linear(probs_F: np.ndarray, probs_GRM: np.ndarray, beta: float) -> np.ndarray:
    """Convex combination in probability space."""
    out = beta * probs_F + (1.0 - beta) * probs_GRM
    return out / out.sum(axis=1, keepdims=True)


def blend_log(probs_F: np.ndarray, probs_GRM: np.ndarray, beta: float) -> np.ndarray:
    """Geometric mean in probability space; convex in log-space (entropy-preserving)."""
    log_F = np.log(np.clip(probs_F, 1e-12, 1.0))
    log_G = np.log(np.clip(probs_GRM, 1e-12, 1.0))
    log_blend = beta * log_F + (1.0 - beta) * log_G
    blend = np.exp(log_blend - log_blend.max(axis=1, keepdims=True))
    return blend / blend.sum(axis=1, keepdims=True)


def blend_gate(probs_F: np.ndarray, probs_GRM: np.ndarray, threshold: float) -> np.ndarray:
    """If F's top-1 probability ≤ threshold, use F; otherwise fall back to GRM.

    Encodes the calibration finding: F is calibrated in its low-confidence regime
    and catastrophically miscalibrated in its high-confidence regime, so swap to
    the calibrated-but-humble GRM whenever F gets bold.
    """
    use_F = probs_F.max(axis=1) <= threshold
    out = np.where(use_F[:, None], probs_F, probs_GRM)
    return out / out.sum(axis=1, keepdims=True)


def blend_density(
    probs_F: np.ndarray, probs_GRM: np.ndarray, density: np.ndarray, scale: float, shift: float
) -> np.ndarray:
    """Sigmoid-weighted blend driven by feature-space density.

    Per-visit weight on F: σ(scale * (density − shift)). Dense regions → trust F;
    sparse regions → trust GRM.
    """
    w_F = 1.0 / (1.0 + np.exp(-scale * (density - shift)))
    out = w_F[:, None] * probs_F + (1.0 - w_F)[:, None] * probs_GRM
    return out / out.sum(axis=1, keepdims=True)


def compute_feature_density(X: np.ndarray, k: int) -> np.ndarray:
    """Per-row inverse k-NN distance, scaled to [0, 1] across the dataset.

    Density = 1 / (1 + d_k) where d_k is the distance to the k-th nearest neighbor.
    Scale to unit range so the sigmoid `shift` parameter has a consistent meaning.
    """
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_sc)
    d, _ = nn.kneighbors(X_sc)
    d_k = d[:, k]  # distance to k-th neighbor (self is column 0)
    density = 1.0 / (1.0 + d_k)
    lo, hi = float(density.min()), float(density.max())
    if hi - lo < 1e-9:
        return np.zeros_like(density)
    return (density - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Metrics + bootstrap
# ---------------------------------------------------------------------------


def evaluate_probs(
    probs: np.ndarray, y_true: np.ndarray, n_states: int, top_k: int = 2
) -> Dict[str, float]:
    """Compute the four key metrics on full-class probability predictions."""
    safe = np.clip(probs, 1e-12, 1.0)
    safe = safe / safe.sum(axis=1, keepdims=True)
    rows = np.arange(len(y_true))
    log_loss_val = float(-np.mean(np.log(safe[rows, y_true])))
    brier = _multi_brier(y_true, safe, n_states)
    top1 = float(np.mean(safe.argmax(axis=1) == y_true))
    top2 = _topk_accuracy(y_true, safe, k=top_k)
    ece = expected_calibration_error(y_true, safe, n_bins=10)
    return {
        "log_loss": log_loss_val,
        "brier": brier,
        "top1_acc": top1,
        "top2_acc": top2,
        "ece": ece,
    }


def bootstrap_metric_ci(
    metric_fn: callable,
    *,
    subject_ids: np.ndarray,
    n_boot: int,
    seed: int,
    label: str,
) -> Tuple[float, float, float]:
    """Cluster bootstrap a single scalar metric_fn(idx) over subjects."""

    def paired(idx: np.ndarray) -> Tuple[float, ...]:
        return (metric_fn(idx),)

    res = cluster_bootstrap_paired(subject_ids, paired, n_boot=n_boot, seed=seed, label=label)
    return res[0]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def build_candidate_grid(density: np.ndarray) -> List[Tuple[str, Dict[str, Any]]]:
    """Enumerate (name, params) for every blend candidate."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    out.append(("reference_F", {"kind": "reference_F"}))
    out.append(("reference_GRM", {"kind": "reference_GRM"}))
    out.append(("reference_A_markov", {"kind": "reference_A_markov"}))
    out.append(("reference_B_markov_laplace", {"kind": "reference_B_markov_laplace"}))
    out.append(("linear_beta_0.50", {"kind": "linear", "beta": 0.5}))

    for beta in np.linspace(0.0, 1.0, 11):
        out.append((f"linear_beta_{beta:.2f}", {"kind": "linear", "beta": float(beta)}))
    for beta in np.linspace(0.0, 1.0, 11):
        out.append((f"log_beta_{beta:.2f}", {"kind": "log", "beta": float(beta)}))
    for thr in np.linspace(0.0, 1.0, 21):
        out.append((f"gate_threshold_{thr:.2f}", {"kind": "gate", "threshold": float(thr)}))
    for scale in [1.0, 2.0, 5.0, 10.0]:
        for shift in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            out.append((
                f"density_scale_{scale:.1f}_shift_{shift:+.2f}",
                {"kind": "density", "scale": float(scale), "shift": float(shift)},
            ))
    return out


def apply_blend(name: str, params: Dict[str, Any], data: Dict[str, Any], density: np.ndarray) -> np.ndarray:
    """Compute per-visit probability matrix for a single blend candidate."""
    kind = params["kind"]
    F = data["probs_F"]
    G = data["probs_GRM"]
    if kind == "reference_F":
        return F
    if kind == "reference_GRM":
        return G
    if kind == "reference_A_markov":
        return data["probs_A_markov"]
    if kind == "reference_B_markov_laplace":
        return data["probs_B_markov_laplace"]
    if kind == "linear":
        return blend_linear(F, G, params["beta"])
    if kind == "log":
        return blend_log(F, G, params["beta"])
    if kind == "gate":
        return blend_gate(F, G, params["threshold"])
    if kind == "density":
        return blend_density(F, G, density, params["scale"], params["shift"])
    raise ValueError(f"Unknown blend kind: {kind}")


def evaluate_candidate(
    name: str,
    params: Dict[str, Any],
    data: Dict[str, Any],
    density: np.ndarray,
    cfg: AutotuneConfig,
) -> Dict[str, Any]:
    """Compute metrics + cluster-bootstrap CIs for one blend candidate."""
    probs = apply_blend(name, params, data, density)
    metrics = evaluate_probs(probs, data["y_true"], data["n_states"])

    subj = data["subject_id"]
    y = data["y_true"]
    n_states = data["n_states"]

    def boot_log_loss(idx: np.ndarray) -> float:
        if idx.size == 0:
            return float("nan")
        safe = np.clip(probs[idx], 1e-12, 1.0)
        rows = np.arange(len(idx))
        return float(-np.mean(np.log(safe[rows, y[idx]])))

    def boot_ece(idx: np.ndarray) -> float:
        if idx.size == 0:
            return float("nan")
        return expected_calibration_error(y[idx], probs[idx], n_bins=10)

    ll_pt, ll_lo, ll_hi = bootstrap_metric_ci(
        boot_log_loss, subject_ids=subj, n_boot=cfg.bootstrap_n, seed=cfg.seed, label=f"{name}:log_loss",
    )
    ece_pt, ece_lo, ece_hi = bootstrap_metric_ci(
        boot_ece, subject_ids=subj, n_boot=cfg.bootstrap_n, seed=cfg.seed, label=f"{name}:ece",
    )
    return {
        "candidate": name,
        **{f"param_{k}": v for k, v in params.items()},
        **metrics,
        "log_loss_ci_low": ll_lo,
        "log_loss_ci_high": ll_hi,
        "ece_ci_low": ece_lo,
        "ece_ci_high": ece_hi,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_log_loss_vs_param(leaderboard: pd.DataFrame, plot_dir: Path) -> None:
    """One panel per blend family showing log-loss as the swept parameter varies."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, kind, x_col, x_label in [
        (axes[0], "linear", "param_beta", "β (weight on F)"),
        (axes[1], "log", "param_beta", "β (weight on F, log-space)"),
        (axes[2], "gate", "param_threshold", "τ (F max-prob threshold)"),
    ]:
        sub = leaderboard[leaderboard["param_kind"] == kind].sort_values(x_col)
        if sub.empty:
            ax.set_visible(False)
            continue
        x = sub[x_col].to_numpy(float)
        y = sub["log_loss"].to_numpy(float)
        err_lo = y - sub["log_loss_ci_low"].to_numpy(float)
        err_hi = sub["log_loss_ci_high"].to_numpy(float) - y
        ax.errorbar(x, y, yerr=[err_lo, err_hi], marker="o", capsize=3)
        ref_F = float(leaderboard[leaderboard["candidate"] == "reference_F"]["log_loss"].iloc[0])
        ref_GRM = float(leaderboard[leaderboard["candidate"] == "reference_GRM"]["log_loss"].iloc[0])
        ax.axhline(ref_F, linestyle="--", linewidth=1, label="F alone")
        ax.axhline(ref_GRM, linestyle=":", linewidth=1, label="GRM alone")
        ax.set_xlabel(x_label)
        ax.set_title(f"{kind} blend")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("log-loss (lower = better)")
    fig.suptitle("Transition log-loss vs blend parameter (subject-CV, bootstrap 95% CI)")
    _save(fig, plot_dir / "log_loss_vs_param.png")


def plot_log_loss_vs_ece_pareto(leaderboard: pd.DataFrame, plot_dir: Path) -> None:
    """Scatter of log-loss vs ECE for all candidates; highlight Pareto-optimal points."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ll = leaderboard["log_loss"].to_numpy(float)
    ece = leaderboard["ece"].to_numpy(float)
    is_ref = leaderboard["param_kind"].astype(str).str.startswith("reference")
    ax.scatter(ll[~is_ref], ece[~is_ref], s=14, alpha=0.5, label="blend candidates")
    ax.scatter(ll[is_ref], ece[is_ref], s=60, marker="*", label="references (F, GRM, A, B)")
    pareto = pareto_front_indices(ll, ece)
    ax.scatter(ll[pareto], ece[pareto], s=80, facecolors="none", edgecolors="black", linewidths=1.2, label="Pareto front")
    for i in pareto:
        ax.annotate(leaderboard["candidate"].iloc[i], (ll[i], ece[i]), fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("log-loss (lower = better)")
    ax.set_ylabel("ECE (lower = better)")
    ax.set_title("Pareto front: log-loss vs ECE across blend candidates")
    ax.legend(fontsize=8, loc="best")
    _save(fig, plot_dir / "log_loss_vs_ece_pareto.png")


def pareto_front_indices(loss: np.ndarray, ece: np.ndarray) -> np.ndarray:
    """Return indices of points NOT dominated on (loss, ece) — lower is better on both."""
    n = len(loss)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated_by_j = (loss <= loss[i]) & (ece <= ece[i]) & ((loss < loss[i]) | (ece < ece[i]))
        if dominated_by_j.any():
            keep[i] = False
    return np.where(keep)[0]


def plot_reliability_top3(
    leaderboard: pd.DataFrame, data: Dict[str, Any], density: np.ndarray, plot_dir: Path,
) -> None:
    """Reliability curves for the top-3 candidates by log-loss plus F and GRM references."""
    best = leaderboard.sort_values("log_loss").head(3)
    refs = leaderboard[leaderboard["candidate"].isin(["reference_F", "reference_GRM"])]
    chosen = pd.concat([best, refs]).drop_duplicates(subset=["candidate"])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    for _, row in chosen.iterrows():
        params = {k.replace("param_", ""): v for k, v in row.items() if k.startswith("param_") and pd.notna(v)}
        probs = apply_blend(row["candidate"], params, data, density)
        rel = reliability_table(data["y_true"], probs)
        rel = rel[rel["n"] > 0]
        ax.plot(rel["mean_confidence"], rel["empirical_accuracy"], marker="o", label=row["candidate"])
    ax.set_xlabel("Mean predicted top-1 probability")
    ax.set_ylabel("Empirical top-1 accuracy")
    ax.set_title("Reliability — top-3 blends + F + GRM references")
    ax.legend(fontsize=8, loc="best")
    _save(fig, plot_dir / "reliability_top3.png")


# ---------------------------------------------------------------------------
# Summary + main
# ---------------------------------------------------------------------------


def best_per_kind(leaderboard: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Return the lowest-log_loss candidate for each blend family."""
    out: Dict[str, Dict[str, Any]] = {}
    for kind in leaderboard["param_kind"].dropna().unique():
        sub = leaderboard[leaderboard["param_kind"] == kind]
        winner = sub.sort_values("log_loss").iloc[0]
        out[str(kind)] = {col: (None if pd.isna(winner[col]) else winner[col]) for col in winner.index}
    return out


def print_top(leaderboard: pd.DataFrame, n: int = 10) -> None:
    """Print the top-n candidates by log-loss for quick visual confirmation."""
    head = leaderboard.sort_values("log_loss").head(n)
    print()
    print("=" * 78)
    print(f"Top {n} candidates by log-loss")
    print("=" * 78)
    print(f"{'rank':>4} {'candidate':40s} {'log_loss':>9} {'95% CI':>17} {'ECE':>7}")
    for r, (_, row) in enumerate(head.iterrows(), start=1):
        ci = f"[{row['log_loss_ci_low']:.3f},{row['log_loss_ci_high']:.3f}]"
        print(f"{r:>4} {row['candidate']:40s} {row['log_loss']:>9.4f} {ci:>17} {row['ece']:>7.4f}")


def write_manifest(cfg: AutotuneConfig) -> Dict[str, Any]:
    """Write a JSON manifest with config + git provenance."""
    manifest = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "git_commit": git_commit_info(),
        "tier": "eval-blending-only",
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def main() -> None:
    """Run the Tier-1 autotune end-to-end."""
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(cfg)

    print("[1/4] Collecting per-visit predictions via subject-CV")
    data = collect_per_visit_predictions(cfg)

    print(f"[2/4] Computing feature density (k={cfg.density_k})")
    density = compute_feature_density(data["baseline_features"], k=cfg.density_k)

    print("[3/4] Evaluating blend grid")
    candidates = build_candidate_grid(density)
    print(f"  {len(candidates)} candidates")
    rows: List[Dict[str, Any]] = []
    for i, (name, params) in enumerate(candidates, start=1):
        if i % 20 == 0 or i == len(candidates):
            print(f"  evaluating {i}/{len(candidates)}: {name}")
        rows.append(evaluate_candidate(name, params, data, density, cfg))
    leaderboard = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    leaderboard.to_csv(cfg.output_dir / "leaderboard.csv", index=False)

    with open(cfg.output_dir / "best_per_kind.json", "w", encoding="utf-8") as f:
        json.dump(best_per_kind(leaderboard), f, indent=2, default=str)

    pareto = pareto_front_indices(
        leaderboard["log_loss"].to_numpy(float), leaderboard["ece"].to_numpy(float),
    )
    leaderboard.iloc[pareto].sort_values("log_loss").to_csv(
        cfg.output_dir / "pareto_front.csv", index=False,
    )

    print("[4/4] Generating plots")
    plot_dir = cfg.output_dir / "plots"
    plot_log_loss_vs_param(leaderboard, plot_dir)
    plot_log_loss_vs_ece_pareto(leaderboard, plot_dir)
    plot_reliability_top3(leaderboard, data, density, plot_dir)

    # Cache per-visit predictions for downstream Tier 2 / Tier 3 runs.
    pred_df = pd.DataFrame({
        "subject_id": data["subject_id"],
        "y_true": data["y_true"],
        "current_state": data["current_state"],
        "end_day": data["end_days"],
        "density": density,
    })
    for j in range(data["probs_F"].shape[1]):
        pred_df[f"P_F_{j}"] = data["probs_F"][:, j]
        pred_df[f"P_GRM_{j}"] = data["probs_GRM"][:, j]
    pred_df.to_parquet(cfg.output_dir / "per_visit_predictions.parquet", index=False)

    print_top(leaderboard, n=10)
    print()
    print(f"Outputs written to: {cfg.output_dir.resolve()}")
    print("Files: leaderboard.csv, best_per_kind.json, pareto_front.csv, manifest.json, "
          "per_visit_predictions.parquet, plots/*.png")
    print()
    print(
        "Scientific framing: this is a search procedure on a synthetic benchmark. A 'winning' "
        "blend here is evidence that, on this dataset, GRM and F have complementary failure "
        "modes — not evidence about clinical performance."
    )


if __name__ == "__main__":
    main()
