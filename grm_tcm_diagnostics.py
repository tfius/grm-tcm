from __future__ import annotations

"""
Diagnostics for the synthetic GRM-TCM latent-state benchmark.

This script evaluates whether the GRM embeddings recovered useful latent
structure in the synthetic benchmark. It is diagnostic only: high scores are
evidence of synthetic latent-state recovery and mismatch detection, not proof of
TCM, Qi, or a biological mechanism.

Run:
  python grm_tcm_diagnostics.py
"""

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass
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
        "matplotlib is required for diagnostics plots. Run `uv sync` or execute `uv run python grm_tcm_diagnostics.py`."
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    roc_auc_score,
)

from grm_tcm_plot_captions import save_with_caption


LATENT_NAMES = [
    "vitality_depletion",
    "stress_activation",
    "inflammatory_load",
    "digestive_instability",
]

GROUP_COLUMNS = [
    "hidden_subtype",
    "true_regime",
    "attractor_state",
    "qi_like_label",
    "tcm_like_label",
    "contrarian_signature",
]

CLUSTER_TARGETS = [
    "hidden_subtype",
    "true_regime",
    "true_regime_id",
    "attractor_state",
    "qi_like_label",
    "tcm_like_label",
    "contrarian_signature",
]


@dataclass
class DiagnosticsConfig:
    """Filesystem configuration for diagnostics inputs and outputs."""

    data_dir: str = "synthetic_grm_tcm"
    results_dir: str = "grm_tcm_results"
    output_dir: str = "grm_tcm_diagnostics"


def parse_args() -> DiagnosticsConfig:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run GRM-TCM synthetic diagnostics.")
    parser.add_argument("--data-dir", default="synthetic_grm_tcm")
    parser.add_argument("--results-dir", default="grm_tcm_results")
    parser.add_argument("--output-dir", default="grm_tcm_diagnostics")
    args = parser.parse_args()
    return DiagnosticsConfig(data_dir=args.data_dir, results_dir=args.results_dir, output_dir=args.output_dir)


def read_csv_optional(path: Path, required: bool = False) -> Optional[pd.DataFrame]:
    """Read a CSV file, returning None for missing optional files."""

    if not path.exists():
        message = f"Missing {'required' if required else 'optional'} file: {path}"
        if required:
            raise FileNotFoundError(message)
        print(f"[skip] {message}")
        return None
    print(f"[load] {path}")
    return pd.read_csv(path)


def read_json_optional(path: Path) -> Dict[str, Any]:
    """Read a JSON file, returning an empty dict if it is missing."""

    if not path.exists():
        print(f"[skip] Missing optional file: {path}")
        return {}
    print(f"[load] {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mode_columns(df: pd.DataFrame) -> List[str]:
    """Return GRM mode columns in numeric order."""

    cols = [c for c in df.columns if c.startswith("grm_mode_")]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[1]))


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""

    path.mkdir(parents=True, exist_ok=True)


def merge_analysis_frame(
    visits: pd.DataFrame,
    latent: Optional[pd.DataFrame],
    subjects: Optional[pd.DataFrame],
    events: Optional[pd.DataFrame],
    embeddings: Optional[pd.DataFrame],
    predictions: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge synthetic truth, GRM embeddings, predictions, and subject metadata."""

    df = visits.copy()
    if latent is not None:
        latent_truth = latent.rename(columns={name: f"true_{name}" for name in LATENT_NAMES if name in latent.columns})
        keep_cols = ["subject_id", "day"] + [f"true_{name}" for name in LATENT_NAMES if f"true_{name}" in latent_truth.columns]
        df = df.merge(latent_truth[keep_cols], on=["subject_id", "day"], how="left")
    else:
        for name in LATENT_NAMES:
            if name in df.columns:
                df[f"true_{name}"] = df[name]

    if subjects is not None:
        df = df.merge(subjects, on="subject_id", how="left")

    if events is not None and not events.empty and {"subject_id", "day", "event_type"}.issubset(events.columns):
        event_counts = pd.crosstab([events["subject_id"], events["day"]], events["event_type"]).reset_index()
        event_counts.columns.name = None
        df = df.merge(event_counts, on=["subject_id", "day"], how="left")
        event_cols = [c for c in event_counts.columns if c not in {"subject_id", "day"}]
        df[event_cols] = df[event_cols].fillna(0).astype(int)

    if embeddings is not None:
        embed_cols = ["visit_id", "subject_id", "day"] + mode_columns(embeddings)
        if "visit_id" in df.columns:
            df = df.merge(embeddings[embed_cols], on=["visit_id", "subject_id", "day"], how="left")
        else:
            df = df.merge(embeddings[embed_cols], on=["subject_id", "day"], how="left")

    if predictions is not None:
        pred_cols = [c for c in predictions.columns if c not in {"subject_id", "day"}]
        if "visit_id" in df.columns and "visit_id" in predictions.columns:
            df = df.merge(predictions[pred_cols], on="visit_id", how="left", suffixes=("", "_predfile"))
        else:
            df = df.merge(predictions, on=["subject_id", "day"], how="left", suffixes=("", "_predfile"))

        for col in ["next_day_score", "flare_next_day"]:
            pred_col = f"{col}_predfile"
            if pred_col in df.columns and col in df.columns:
                df[col] = df[col].combine_first(df[pred_col])
                df = df.drop(columns=[pred_col])

    return df


def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    """Compute a guarded Pearson correlation."""

    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def latent_correlation_tables(df: pd.DataFrame, modes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build GRM-mode to true-latent correlation diagnostics."""

    latent_cols = [f"true_{name}" for name in LATENT_NAMES if f"true_{name}" in df.columns]
    rows = []
    for mode in modes:
        for latent in latent_cols:
            corr = pearson_corr(df[mode], df[latent])
            rows.append(
                {
                    "grm_mode": mode,
                    "latent_variable": latent.removeprefix("true_"),
                    "correlation": corr,
                    "abs_correlation": abs(corr) if not math.isnan(corr) else np.nan,
                }
            )
    corr_df = pd.DataFrame(rows)
    mode_top = (
        corr_df.sort_values(["grm_mode", "abs_correlation"], ascending=[True, False])
        .groupby("grm_mode", as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
    latent_top = (
        corr_df.sort_values(["latent_variable", "abs_correlation"], ascending=[True, False])
        .groupby("latent_variable", as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
    return corr_df, mode_top, latent_top


def regression_stats(df: pd.DataFrame, pred_col: str = "pred_grm_next_score") -> Dict[str, float]:
    """Compute next-day score prediction error statistics."""

    if "next_day_score" not in df.columns or pred_col not in df.columns:
        return {}
    pair = df[["next_day_score", pred_col]].dropna()
    if pair.empty:
        return {}
    err = pair[pred_col] - pair["next_day_score"]
    return {
        "n": int(len(pair)),
        "mae": float(mean_absolute_error(pair["next_day_score"], pair[pred_col])),
        "rmse": float(np.sqrt(mean_squared_error(pair["next_day_score"], pair[pred_col]))),
        "bias": float(err.mean()),
        "mean_error": float(err.mean()),
        "median_abs_error": float(np.median(np.abs(err))),
        "error_std": float(err.std(ddof=0)),
    }


def flare_stats(df: pd.DataFrame, prob_col: str = "pred_grm_flare_prob") -> Dict[str, float]:
    """Compute guarded flare prediction statistics."""

    if "flare_next_day" not in df.columns or prob_col not in df.columns:
        return {}
    pair = df[["flare_next_day", prob_col]].dropna()
    if pair.empty or pair["flare_next_day"].nunique() < 2:
        return {"n": int(len(pair)), "positive_rate": float(pair["flare_next_day"].mean()) if len(pair) else float("nan")}
    y_true = pair["flare_next_day"].astype(int)
    y_prob = pair[prob_col].astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "n": int(len(pair)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def grouped_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute prediction performance grouped by semantic and hidden labels."""

    rows: List[Dict[str, Any]] = []
    for group_col in GROUP_COLUMNS:
        if group_col not in df.columns:
            continue
        for value, group in df.groupby(group_col, dropna=False):
            reg = regression_stats(group)
            flare = flare_stats(group)
            row = {"group_column": group_col, "group_value": str(value), "n_rows": int(len(group))}
            row.update({f"next_day_{k}": v for k, v in reg.items()})
            row.update({f"flare_{k}": v for k, v in flare.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def entropy_from_counts(counts: Iterable[int]) -> Tuple[float, float]:
    """Return entropy and normalized entropy for a count vector."""

    arr = np.asarray(list(counts), dtype=float)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return 0.0, 0.0
    probs = arr / arr.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    normalized = float(entropy / np.log2(len(arr))) if len(arr) > 1 else 0.0
    return entropy, normalized


def distribution_table(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    """Build counts and row percentages for one categorical distribution."""

    if row_col not in df.columns or col_col not in df.columns:
        return pd.DataFrame()
    counts = pd.crosstab(df[row_col], df[col_col])
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    rows = []
    for row_value in counts.index:
        entropy, norm_entropy = entropy_from_counts(counts.loc[row_value].to_numpy())
        for col_value in counts.columns:
            rows.append(
                {
                    "distribution": f"{row_col}_to_{col_col}",
                    row_col: row_value,
                    col_col: col_value,
                    "count": int(counts.loc[row_value, col_value]),
                    "row_fraction": float(pct.loc[row_value, col_value]),
                    "entropy": entropy,
                    "normalized_entropy": norm_entropy,
                }
            )
    return pd.DataFrame(rows)


def ontology_mismatch(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect label/subtype mismatch and possible contrarian findings."""

    required = {"tcm_like_label", "hidden_subtype"}
    if not required.issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame()

    label_to_subtype = distribution_table(df, "tcm_like_label", "hidden_subtype")
    subtype_to_label = distribution_table(df, "hidden_subtype", "tcm_like_label")
    mismatch = pd.concat([label_to_subtype, subtype_to_label], ignore_index=True)

    findings: List[Dict[str, Any]] = []
    label_counts = pd.crosstab(df["tcm_like_label"], df["hidden_subtype"])
    for label in label_counts.index:
        counts = label_counts.loc[label]
        nonzero = counts[counts > 0]
        entropy, norm_entropy = entropy_from_counts(counts.to_numpy())
        if len(nonzero) > 1:
            findings.append(
                {
                    "finding_type": "label_contains_multiple_hidden_subtypes",
                    "label": label,
                    "hidden_subtype": "",
                    "n_categories": int(len(nonzero)),
                    "n_rows": int(counts.sum()),
                    "entropy": entropy,
                    "normalized_entropy": norm_entropy,
                    "detail": json.dumps({str(k): int(v) for k, v in nonzero.items()}),
                }
            )

    subtype_counts = pd.crosstab(df["hidden_subtype"], df["tcm_like_label"])
    for subtype in subtype_counts.index:
        counts = subtype_counts.loc[subtype]
        nonzero = counts[counts > 0]
        entropy, norm_entropy = entropy_from_counts(counts.to_numpy())
        if len(nonzero) > 1:
            findings.append(
                {
                    "finding_type": "hidden_subtype_split_across_labels",
                    "label": "",
                    "hidden_subtype": subtype,
                    "n_categories": int(len(nonzero)),
                    "n_rows": int(counts.sum()),
                    "entropy": entropy,
                    "normalized_entropy": norm_entropy,
                    "detail": json.dumps({str(k): int(v) for k, v in nonzero.items()}),
                }
            )

    findings_df = pd.DataFrame(findings)
    if not findings_df.empty:
        findings_df = findings_df.sort_values(["normalized_entropy", "n_rows"], ascending=[False, False])
    return mismatch, findings_df


def safe_label_codes(series: pd.Series) -> Tuple[np.ndarray, bool]:
    """Factorize labels and report whether at least two classes exist."""

    clean = series.astype("string").fillna("<missing>")
    codes, uniques = pd.factorize(clean)
    return codes, len(uniques) > 1


def cluster_embeddings(df: pd.DataFrame, modes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster GRM embeddings and compare clusters to hidden and semantic labels."""

    if not modes:
        return pd.DataFrame(), pd.DataFrame()
    work = df[["subject_id", "day"] + (["visit_id"] if "visit_id" in df.columns else []) + modes].dropna(subset=modes).copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    scores: List[Dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in [3, 4, 5]:
            if len(work) < k:
                continue
            labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(work[modes].to_numpy(float))
            work[f"cluster_k{k}"] = labels
            source_idx = work.index
            for target in CLUSTER_TARGETS:
                if target not in df.columns:
                    continue
                target_codes, usable = safe_label_codes(df.loc[source_idx, target])
                if not usable:
                    scores.append({"k": k, "target": target, "adjusted_rand_index": np.nan, "normalized_mutual_info": np.nan})
                    continue
                scores.append(
                    {
                        "k": k,
                        "target": target,
                        "adjusted_rand_index": float(adjusted_rand_score(target_codes, labels)),
                        "normalized_mutual_info": float(normalized_mutual_info_score(target_codes, labels)),
                    }
                )
    return work, pd.DataFrame(scores)


def regime_label_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """Compare true regimes against noisy semantic labels and hidden subtype."""

    frames = []
    for col in ["tcm_like_label", "qi_like_label", "hidden_subtype"]:
        if "true_regime" in df.columns and col in df.columns:
            frames.append(distribution_table(df, "true_regime", col))
            frames.append(distribution_table(df, col, "true_regime"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def regime_recovery_summary(cluster_scores: pd.DataFrame) -> Dict[str, float]:
    """Summarize best cluster alignment against true regime targets."""

    out = {
        "best_cluster_nmi_true_regime": float("nan"),
        "best_cluster_ari_true_regime": float("nan"),
        "best_cluster_nmi_attractor_state": float("nan"),
        "best_cluster_ari_attractor_state": float("nan"),
    }
    if cluster_scores.empty:
        return out
    for target, nmi_key, ari_key in [
        ("true_regime", "best_cluster_nmi_true_regime", "best_cluster_ari_true_regime"),
        ("attractor_state", "best_cluster_nmi_attractor_state", "best_cluster_ari_attractor_state"),
    ]:
        subset = cluster_scores[cluster_scores["target"] == target]
        if not subset.empty:
            out[nmi_key] = float(subset["normalized_mutual_info"].max())
            out[ari_key] = float(subset["adjusted_rand_index"].max())
    return out


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Write pretty JSON."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_table(path: Path, df: pd.DataFrame) -> None:
    """Write a CSV table, including an empty CSV for unavailable diagnostics."""

    df.to_csv(path, index=False)
    print(f"[write] {path}")


def matrix_from_correlations(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot correlation rows into a mode by latent-variable matrix."""

    if corr_df.empty:
        return pd.DataFrame()
    return corr_df.pivot(index="grm_mode", columns="latent_variable", values="correlation")


def save_heatmap(matrix: pd.DataFrame, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    """Save a matplotlib heatmap using default styling."""

    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix.to_numpy(float), aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), labels=matrix.index)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_with_caption(fig, path, dpi=160)
    print(f"[plot] {path}")


def save_prediction_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    """Save predicted-vs-actual and residual plots for next-day score."""

    if not {"next_day_score", "pred_grm_next_score"}.issubset(df.columns):
        return
    pair = df[["next_day_score", "pred_grm_next_score"]].dropna()
    if pair.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pair["next_day_score"], pair["pred_grm_next_score"], s=10, alpha=0.6)
    lo = float(min(pair["next_day_score"].min(), pair["pred_grm_next_score"].min()))
    hi = float(max(pair["next_day_score"].max(), pair["pred_grm_next_score"].max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_title("Predicted vs Actual Next-Day Score")
    ax.set_xlabel("Actual next_day_score")
    ax.set_ylabel("Predicted next_day_score")
    fig.tight_layout()
    path = plot_dir / "predicted_vs_actual_next_day_score.png"
    save_with_caption(fig, path, dpi=160)
    print(f"[plot] {path}")

    residual = pair["pred_grm_next_score"] - pair["next_day_score"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residual, bins=40)
    ax.set_title("Next-Day Score Residuals")
    ax.set_xlabel("Prediction residual")
    ax.set_ylabel("Count")
    fig.tight_layout()
    path = plot_dir / "residual_histogram.png"
    save_with_caption(fig, path, dpi=160)
    print(f"[plot] {path}")


def save_mode_scatter(df: pd.DataFrame, label_col: str, path: Path) -> None:
    """Save first-two-mode scatter plot colored by a categorical label."""

    required = {"grm_mode_1", "grm_mode_2", label_col}
    if not required.issubset(df.columns):
        return
    plot_df = df[["grm_mode_1", "grm_mode_2", label_col]].dropna()
    if plot_df.empty or plot_df[label_col].nunique() < 1:
        return
    codes, uniques = pd.factorize(plot_df[label_col].astype(str))
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(plot_df["grm_mode_1"], plot_df["grm_mode_2"], c=codes, s=10, alpha=0.65)
    handles, _ = scatter.legend_elements()
    labels = [str(x) for x in uniques[: len(handles)]]
    if handles:
        ax.legend(handles, labels, title=label_col, loc="best", fontsize="small")
    ax.set_title(f"GRM Modes 1-2 by {label_col}")
    ax.set_xlabel("grm_mode_1")
    ax.set_ylabel("grm_mode_2")
    fig.tight_layout()
    save_with_caption(fig, path, dpi=160)
    print(f"[plot] {path}")


def save_group_mean_modes(df: pd.DataFrame, modes: List[str], group_col: str, path: Path) -> None:
    """Save mean GRM mode values by group as a matrix plot."""

    if group_col not in df.columns or not modes:
        return
    grouped = df.groupby(group_col, dropna=False)[modes].mean().dropna(how="all")
    if grouped.empty:
        return
    save_heatmap(grouped, path, f"Mean GRM Modes by {group_col}", "GRM mode", group_col)


def save_regime_plots(df: pd.DataFrame, plot_dir: Path) -> None:
    """Save true-regime occupancy and label mismatch plots."""

    if {"hidden_subtype", "true_regime"}.issubset(df.columns):
        counts = pd.crosstab(df["hidden_subtype"], df["true_regime"], normalize="index")
        save_heatmap(counts, plot_dir / "true_regime_occupancy_by_hidden_subtype.png", "True Regime Occupancy by Hidden Subtype", "true_regime", "hidden_subtype")

    if {"tcm_like_label", "true_regime"}.issubset(df.columns):
        counts = pd.crosstab(df["tcm_like_label"], df["true_regime"], normalize="index")
        save_heatmap(counts, plot_dir / "true_regime_distribution_by_tcm_label.png", "True Regime Distribution by TCM-Like Label", "true_regime", "tcm_like_label")


def save_manifold_scatter_3d(df: pd.DataFrame, plot_dir: Path) -> None:
    """3D scatter of the first three GRM modes, colored by true_regime and hidden_subtype."""

    needed = ["grm_mode_1", "grm_mode_2", "grm_mode_3"]
    if not set(needed).issubset(df.columns):
        return
    label_cols = [c for c in ("true_regime", "hidden_subtype") if c in df.columns]
    if not label_cols:
        return
    plot_df = df[needed + label_cols].dropna(subset=needed)
    if plot_df.empty:
        return

    fig = plt.figure(figsize=(12, 5))
    for i, label_col in enumerate(label_cols, start=1):
        subdf = plot_df.dropna(subset=[label_col])
        if subdf.empty:
            continue
        codes, uniques = pd.factorize(subdf[label_col].astype(str))
        ax = fig.add_subplot(1, len(label_cols), i, projection="3d")
        scatter = ax.scatter(
            subdf["grm_mode_1"],
            subdf["grm_mode_2"],
            subdf["grm_mode_3"],
            c=codes,
            s=6,
            alpha=0.55,
        )
        ax.set_xlabel("grm_mode_1")
        ax.set_ylabel("grm_mode_2")
        ax.set_zlabel("grm_mode_3")
        ax.set_title(f"Modes 1-3 by {label_col}")
        handles, _ = scatter.legend_elements()
        labels = [str(x) for x in uniques[: len(handles)]]
        if handles:
            ax.legend(handles, labels, title=label_col, loc="best", fontsize="x-small")

    fig.tight_layout()
    path = plot_dir / "manifold_scatter_3d.png"
    save_with_caption(fig, path, dpi=160)
    print(f"[plot] {path}")


def save_eigen_spectrum(results_dir: Path, plot_dir: Path) -> None:
    """Plot graph-Laplacian eigenvalues with the n_modes cutoff annotated."""

    basis_path = results_dir / "model" / "grm_basis.npz"
    if not basis_path.exists():
        print(f"[skip] {basis_path}")
        return
    data = np.load(basis_path)
    # Prefer eigenvalues_full (computed past the cutoff) when available so the
    # plot can show tail behavior; otherwise fall back to the retained set.
    if "eigenvalues_full" in data.files:
        eigenvalues = data["eigenvalues_full"].astype(float)
    else:
        eigenvalues = data["eigenvalues"].astype(float)
    if eigenvalues.size == 0:
        return
    n_modes = int(data["n_modes"]) if "n_modes" in data.files else len(eigenvalues)
    idx = np.arange(1, len(eigenvalues) + 1)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11, 4))
    ax_lin.plot(idx, eigenvalues, marker="o", markersize=3, linewidth=1)
    ax_lin.axvline(n_modes, linestyle="--", linewidth=1)
    ax_lin.set_title("Eigenvalues (linear)")
    ax_lin.set_xlabel("Mode index")
    ax_lin.set_ylabel("lambda_k")

    positive = eigenvalues[eigenvalues > 0]
    floor = float(positive.min()) if positive.size else 1e-8
    clipped = np.clip(eigenvalues, floor, None)
    ax_log.semilogy(idx, clipped, marker="o", markersize=3, linewidth=1)
    ax_log.axvline(n_modes, linestyle="--", linewidth=1)
    ax_log.set_title("Eigenvalues (log)")
    ax_log.set_xlabel("Mode index")
    ax_log.set_ylabel("lambda_k (log)")

    fig.suptitle(f"Graph Laplacian spectrum (n_modes cutoff = {n_modes})")
    fig.tight_layout()
    path = plot_dir / "graph_eigen_spectrum.png"
    save_with_caption(fig, path, dpi=160)
    print(f"[plot] {path}")


def generate_plots(df: pd.DataFrame, corr_df: pd.DataFrame, modes: List[str], output_dir: Path, results_dir: Path) -> None:
    """Generate all required matplotlib plots."""

    plot_dir = output_dir / "plots"
    ensure_dir(plot_dir)
    save_heatmap(
        matrix_from_correlations(corr_df),
        plot_dir / "grm_latent_correlation_heatmap.png",
        "GRM Mode vs True Latent Correlation",
        "True latent variable",
        "GRM mode",
    )
    save_prediction_plots(df, plot_dir)
    save_mode_scatter(df, "hidden_subtype", plot_dir / "grm_modes_scatter_hidden_subtype.png")
    save_mode_scatter(df, "true_regime", plot_dir / "grm_modes_scatter_true_regime.png")
    save_mode_scatter(df, "tcm_like_label", plot_dir / "grm_modes_scatter_tcm_like_label.png")
    save_group_mean_modes(df, modes, "hidden_subtype", plot_dir / "mean_grm_modes_by_hidden_subtype.png")
    save_group_mean_modes(df, modes, "true_regime", plot_dir / "mean_grm_modes_by_true_regime.png")
    save_group_mean_modes(df, modes, "tcm_like_label", plot_dir / "mean_grm_modes_by_tcm_like_label.png")
    save_regime_plots(df, plot_dir)
    save_manifold_scatter_3d(df, plot_dir)
    save_eigen_spectrum(results_dir, plot_dir)


def make_summary(
    cfg: DiagnosticsConfig,
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    corr_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    findings_df: pd.DataFrame,
    cluster_scores: pd.DataFrame,
) -> Dict[str, Any]:
    """Create a compact JSON summary."""

    best_abs_corr = float(corr_df["abs_correlation"].max()) if not corr_df.empty else float("nan")
    best_cluster_nmi = float(cluster_scores["normalized_mutual_info"].max()) if not cluster_scores.empty else float("nan")
    best_cluster_ari = float(cluster_scores["adjusted_rand_index"].max()) if not cluster_scores.empty else float("nan")
    regime_recovery = regime_recovery_summary(cluster_scores)
    return {
        "config": asdict(cfg),
        "n_analysis_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
        "grm_mode_columns": mode_columns(df),
        "best_abs_grm_latent_correlation": best_abs_corr,
        "next_day_prediction": regression_stats(df),
        "flare_prediction": flare_stats(df),
        "best_cluster_normalized_mutual_info": best_cluster_nmi,
        "best_cluster_adjusted_rand_index": best_cluster_ari,
        **regime_recovery,
        "n_grouped_performance_rows": int(len(grouped_df)),
        "n_contrarian_findings": int(len(findings_df)),
        "trainer_metrics": metrics,
        "interpretation_guardrail": (
            "Diagnostics evaluate synthetic latent-state recovery, ontology mismatch, and benchmark behavior. "
            "They do not prove TCM, Qi, or a biological mechanism."
        ),
    }


def print_readme(output_dir: Path) -> None:
    """Print final guidance for reading diagnostics."""

    print("\nDiagnostics complete.")
    print(f"Outputs written to: {output_dir.resolve()}")
    print("\nInspect first:")
    print("  1. diagnostics_summary.json")
    print("  2. grm_latent_correlations.csv")
    print("  3. contrarian_findings.csv")
    print("  4. cluster_scores.csv")
    print("  5. regime_label_mismatch.csv")
    print("  6. plots/grm_latent_correlation_heatmap.png")
    print("\nInterpretation:")
    print("  - High GRM/latent correlation means the embedding recovered known synthetic state axes.")
    print("  - High ontology mismatch means semantic labels mix or split hidden synthetic subtypes.")
    print("  - High cluster NMI/ARI means embedding clusters align with a target label family.")
    print("  - True-regime cluster scores test recovery of the redesigned switching-state simulator.")
    print("  - Poor prediction despite good latent recovery can mean the recovered modes are not the predictive axes,")
    print("    the baseline uses easier short-horizon information, or the target is dominated by observed raw features.")
    print("  - These are synthetic benchmark diagnostics, not evidence that TCM, Qi, or a biological mechanism is proven.")


def run(cfg: DiagnosticsConfig) -> None:
    """Run the diagnostics workflow."""

    data_dir = Path(cfg.data_dir)
    results_dir = Path(cfg.results_dir)
    output_dir = Path(cfg.output_dir)
    ensure_dir(output_dir)

    print("[start] Loading inputs")
    visits = read_csv_optional(data_dir / "visits.csv", required=True)
    latent = read_csv_optional(data_dir / "latent_states.csv")
    subjects = read_csv_optional(data_dir / "subjects.csv")
    events = read_csv_optional(data_dir / "events.csv")
    embeddings = read_csv_optional(results_dir / "grm_visit_embeddings.csv")
    predictions = read_csv_optional(results_dir / "grm_predictions.csv")
    feature_modes = read_csv_optional(results_dir / "grm_feature_modes.csv")
    metrics = read_json_optional(results_dir / "grm_metrics.json")

    assert visits is not None
    print("[step] Merging analysis frame")
    df = merge_analysis_frame(visits, latent, subjects, events, embeddings, predictions)
    modes = mode_columns(df)

    print("[step] Computing latent correlations")
    corr_df, mode_top, latent_top = latent_correlation_tables(df, modes)

    print("[step] Computing grouped prediction performance")
    grouped_df = grouped_performance(df)

    print("[step] Detecting ontology mismatch")
    mismatch_df, findings_df = ontology_mismatch(df)
    regime_mismatch_df = regime_label_mismatch(df)

    print("[step] Clustering GRM embeddings")
    cluster_assignments, cluster_scores = cluster_embeddings(df, modes)

    print("[step] Writing tables")
    save_table(output_dir / "grm_latent_correlations.csv", corr_df)
    save_table(output_dir / "grm_mode_top_correlations.csv", mode_top)
    save_table(output_dir / "latent_top_correlations.csv", latent_top)
    save_table(output_dir / "grouped_performance.csv", grouped_df)
    save_table(output_dir / "ontology_mismatch.csv", mismatch_df)
    save_table(output_dir / "regime_label_mismatch.csv", regime_mismatch_df)
    save_table(output_dir / "contrarian_findings.csv", findings_df)
    save_table(output_dir / "cluster_assignments.csv", cluster_assignments)
    save_table(output_dir / "cluster_scores.csv", cluster_scores)
    if feature_modes is not None:
        save_table(output_dir / "source_grm_feature_modes.csv", feature_modes)

    print("[step] Generating plots")
    generate_plots(df, corr_df, modes, output_dir, results_dir)

    summary = make_summary(cfg, df, metrics, corr_df, grouped_df, findings_df, cluster_scores)
    summary_path = output_dir / "diagnostics_summary.json"
    save_json(summary_path, summary)
    print(f"[write] {summary_path}")
    print_readme(output_dir)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(parse_args())
