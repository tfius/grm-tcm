from __future__ import annotations

"""
Plot captioning: a single funnel for fig.savefig that embeds an interpretive
caption into the image itself, so shared-standalone PNGs carry their own context.

Two pieces:
- CAPTIONS: dict mapping plot filename stem -> caption text. Edit wording here.
- save_with_caption(fig, path, ...): the replacement for fig.savefig + plt.close.

Captions are pre-wrapped, drawn in italic sans-serif below the axes in a strip
allocated by *resizing* the figure (axes proportions preserved). Missing entries
fall back to a guardrail string with a one-shot warning.
"""

import textwrap
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


_FALLBACK_CAPTION = (
    "Synthetic GRM-TCM benchmark — see README.md / CLAUDE.md for context. "
    "Results describe behavior on a known generator, not clinical or biological truth."
)

_WARNED_STEMS: set[str] = set()


# ---------------------------------------------------------------------------
# Caption registry — keyed by plot filename stem (no extension).
# Each value is 1-3 sentences focused on: what the axes mean, what 'good' looks
# like, and (where load-bearing) a scientific-framing guardrail.
# ---------------------------------------------------------------------------

CAPTIONS: dict[str, str] = {
    # --- grm_tcm_diagnostics.py ---
    "grm_latent_correlation_heatmap": (
        "GRM modes (rows) vs true generative latents (cols), Pearson correlation after orthogonal "
        "Procrustes alignment. Bright off-diagonal cells mean the spectral embedding recovered the "
        "synthetic latent structure. Synthetic benchmark only."
    ),
    "predicted_vs_actual_next_day_score": (
        "Test-set next-day score: GRM-ridge predictions vs ground truth. Points on the diagonal are "
        "calibrated; horizontal compression toward the mean indicates regression to the mean. R^2 lives "
        "in grm_metrics.json."
    ),
    "residual_histogram": (
        "Distribution of (predicted - actual) next-day score on the held-out test split. Centered around "
        "zero = unbiased; left/right shift = systematic over/under-prediction; long tails = heteroskedastic "
        "error."
    ),
    "grm_modes_scatter_hidden_subtype": (
        "First two GRM modes per visit, colored by the true hidden_subtype (0/1/2). Cluster separation "
        "indicates the embedding tracks the real generative subtypes rather than the semantic labels."
    ),
    "grm_modes_scatter_true_regime": (
        "First two GRM modes per visit, colored by the true_regime id from the generator. Clear regime "
        "separation means the embedding aligns with the underlying state vocabulary."
    ),
    "grm_modes_scatter_tcm_like_label": (
        "First two GRM modes per visit, colored by the naive TCM-like label. Compare with the "
        "hidden_subtype scatter: divergence reveals where semantic labels split or merge real subtypes."
    ),
    "mean_grm_modes_by_hidden_subtype": (
        "Per-subtype mean GRM mode values. Rows are subtypes, columns are modes; color encodes the mean. "
        "Rows that are distinct from each other show subtype-discriminative modes."
    ),
    "mean_grm_modes_by_true_regime": (
        "Per-regime mean GRM mode values. Distinct rows indicate modes that discriminate among the true "
        "generative regimes — an upper bound on inferred-state quality."
    ),
    "mean_grm_modes_by_tcm_like_label": (
        "Per-TCM-label mean GRM mode values. Compare row distinctness against the hidden_subtype version "
        "to see whether the semantic labels carve up the embedding the same way the real structure does."
    ),
    "true_regime_occupancy_by_hidden_subtype": (
        "Row-normalized crosstab: fraction of visits in each true_regime, conditioned on hidden_subtype. "
        "Diagonal-dominant rows mean subtypes occupy distinct regimes; off-diagonal mass exposes overlap."
    ),
    "true_regime_distribution_by_tcm_label": (
        "Row-normalized crosstab: fraction of visits in each true_regime, conditioned on tcm_like_label. "
        "Mass spread across columns reveals where naive labels merge multiple true regimes — the "
        "ontology-mismatch signal."
    ),
    # --- grm_tcm_dynamic_grm.py ---
    "state_source_metric_comparison": (
        "Headline dynamic metrics under three state vocabularies: kmeans_observation (default), "
        "kmeans_dynamic (trajectory features), and true_regime (oracle ceiling). Use the gap between "
        "kmeans_* and true_regime to read how much state-vocabulary noise costs."
    ),
    "rolling_regime_change_score": (
        "Frobenius norm ||G^(t) - G^(t-1)|| of the pooled rolling GRM resonance matrix over time. Spikes "
        "indicate the propagator is reorganizing; flat = regime is stable. Compare against event days "
        "to see if changes lead flares or crashes."
    ),
    "selected_modes_over_time": (
        "Number of spectral modes retained per window under the 95% energy rule (capped at max_modes). "
        "Flatness can mean either stable spectra OR saturation at the cap — see the saturation plot."
    ),
    "selected_modes_saturation": (
        "Selected modes vs the uncapped modes needed to reach the 95% energy threshold. Whenever the "
        "uncapped line exceeds max_modes (dashed), the energy rule is being clipped — a flat selected "
        "line then reflects the cap, not the spectrum."
    ),
    "cumulative_spectral_energy": (
        "Cumulative spectral energy vs mode rank for ~8 representative windows. Curves that level off "
        "early have a low effective rank; curves that climb slowly need more modes to explain variance."
    ),
    "mean_cumulative_spectral_energy": (
        "Cumulative spectral energy averaged across all rolling windows. The 95% threshold and max_modes "
        "cap are marked; their intersection tells you whether your cap is well-sized for typical windows."
    ),
    "self_resonance_vs_dysregulation": (
        "Per-visit G_ii (state self-loop strength) vs global_dysregulation_score. A positive trend would "
        "say stuck states co-occur with worse symptoms — a 'sticky attractor' signature. Synthetic only."
    ),
    "subject_regime_change_score": (
        "Subject-conditioned regime change ||G_s^(t) - G_s^(t-1)|| aggregated per day (mean and 90th "
        "percentile). Use this to see whether individual-level dynamics shift before pooled-population "
        "shifts do."
    ),
    "subject_self_resonance_vs_dysregulation": (
        "Subject-restricted G_ii vs global_dysregulation_score. Positive trend = individual-level "
        "stuck-state signature; flat = population-level pattern not reflected per-subject."
    ),
    "soft_self_resonance_vs_dysregulation": (
        "Soft-assigned self-resonance (w · diag(G)) vs dysregulation, using the visit's RBF soft state "
        "weights. Smoother than the hard-state version; useful when state boundaries are uncertain."
    ),
    "pooled_transition_reliability": (
        "Calibration diagram for next-state prediction, pooled across subjects. X = mean predicted top-1 "
        "probability per bin, Y = empirical top-1 accuracy. Curve closer to the diagonal = better-calibrated "
        "probabilities. Compare GRM-blended vs Markov-only."
    ),
    "subject_transition_reliability": (
        "Same as the pooled reliability diagram but built from subject-conditioned G_s^(t). If the subject "
        "curve hugs the diagonal more tightly than the pooled one, per-subject conditioning is buying "
        "calibration."
    ),
    "inferred_state_true_regime_confusion": (
        "Row-normalized confusion between inferred KMeans state_id (rows) and the true_regime label "
        "(cols). Diagonal-dominant rows mean inferred states track real regimes; spread = aliasing."
    ),
    "true_stuck_occupancy_by_hidden_subtype": (
        "Per-hidden_subtype mean fraction of days spent in each true stuck regime "
        "(stuck_depleted, stuck_agitated). Tall bars in one subtype but not others = subtype-specific "
        "attractor preference."
    ),
    "subject_resonance_vs_true_stuck_occupancy": (
        "Per-subject mean soft self-resonance vs fraction of days in any true stuck regime. A positive "
        "trend means GRM self-resonance is a defensible per-subject 'stuckness' proxy on this benchmark."
    ),
    # --- grm_tcm_dynamic_eval.py ---
    "transition_log_loss_by_model": (
        "Log-loss for next-state prediction across model variants (subject-CV, 95% bootstrap CIs). Lower "
        "is better. Differences smaller than the CI overlap are not significant."
    ),
    "transition_reliability": (
        "Single-fold calibration diagram for top-1 next-state predictions. Diagonal = perfect "
        "calibration; below = overconfident; above = underconfident. Compare model curves directly."
    ),
    "subject_fingerprint_macro_f1": (
        "Macro-F1 of recovering hidden_subtype from per-subject feature aggregates, 5-fold stratified. "
        "Bars compare GRM-derived fingerprints against raw-observation and label-based controls; error "
        "bars are fold std."
    ),
    "aliased_visits_scatter": (
        "Two-panel comparison on observation-aliased visits only: raw-observation PCA-2 (left) vs t-SNE "
        "of the FULL multi-mode GRM embedding (right), colored by true_regime. If the right panel "
        "separates regimes that the left blurs, GRM is disambiguating under aliasing using information "
        "spread across all modes — not just modes 1–2."
    ),
    "aliased_mode_pair_grid": (
        "Upper-triangle pairwise scatter of the first few GRM modes on the aliased subset, colored by "
        "true_regime. Locates which mode pair (if any) carries discriminative signal — the rest of the "
        "spectrum may be doing the heavy lifting."
    ),
    "aliased_per_mode_histograms": (
        "Per-mode regime-conditional histograms over aliased visits. Modes whose regime distributions "
        "are visibly displaced are the ones that disentangle states under observation aliasing; modes "
        "with fully overlapping histograms are not contributing to T1 entropy lift."
    ),
    "aliased_nn_entropy_heatmap": (
        "Same obs-PCA layout in both panels; color = regime-NN entropy under observation-NN (left) vs "
        "GRM-embedding-NN (right). Bluer = sharper regime concentration. Spatial regions where the right "
        "panel is bluer than the left are where GRM is winning the T1 disambiguation."
    ),
    "transductive_vs_inductive_metrics": (
        "Side-by-side comparison of headline metrics under transductive (in-graph train/test) vs strict "
        "inductive (subject-disjoint) evaluation on the same dataset. Inductive bars below transductive "
        "bars on the GRM-side metrics measure the graph-leak premium being squeezed out by honest holding."
    ),
    "verdicts_forest": (
        "Forest plot companion to the boxed verdicts table. Each row is one falsifiable claim; the dot is "
        "the point estimate Δ and whiskers are the bootstrap 95% CI. PASS = CI strictly positive, FAIL = "
        "CI strictly negative, MARGINAL = CI crosses zero. Dashed line at zero is the null."
    ),
    "flare_calibration_raw_vs_temperature": (
        "Reliability diagram on the inductive test set: raw GRM-logistic flare probability (left) vs "
        "temperature-calibrated probability (right). Right curve hugging the diagonal more tightly than "
        "left = the persisted temperature is earning its keep. ECE reported in each subtitle."
    ),
    "per_subject_performance": (
        "Per-subject heterogeneity on the inductive test set: box + strip of subject-level next-day R² "
        "(left) and flare AUC (right). Tight box = uniform performance; long whiskers / outliers = "
        "averaging successes with disasters. Subjects need ≥5 test visits to be included."
    ),
    "ablation_attractor_auc": (
        "Attractor-AUC on aliased visits across embedding ablations (full GRM, time-shuffled, random, raw "
        "observations). Dashed line = chance (0.5). The drop from full_grm to ablations measures how "
        "much of the signal depends on the actual GRM structure."
    ),
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _resolve_caption(stem: str, explicit: Optional[str]) -> str:
    if explicit is not None:
        return explicit
    if stem in CAPTIONS:
        return CAPTIONS[stem]
    if stem not in _WARNED_STEMS:
        warnings.warn(
            f"No caption registered for plot stem {stem!r}; using fallback. "
            f"Add an entry to grm_tcm_plot_captions.CAPTIONS.",
            stacklevel=3,
        )
        _WARNED_STEMS.add(stem)
    return _FALLBACK_CAPTION


def _auto_wrap_width(fig: Figure) -> int:
    """Approximate characters that fit on one line at 9pt sans-serif."""

    width_inches = float(fig.get_size_inches()[0])
    # Empirical: ~11 characters per inch for italic 9pt DejaVu Sans.
    return max(40, int(width_inches * 11))


def save_with_caption(
    fig: Figure,
    path: Path,
    *,
    caption: Optional[str] = None,
    dpi: int = 160,
    wrap_width: Optional[int] = None,
) -> None:
    """Save a matplotlib figure with an italic caption strip below the axes.

    The figure is resized vertically to make room — the axes area is preserved.
    Caption defaults to CAPTIONS[path.stem] with a one-shot warning fallback.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _resolve_caption(path.stem, caption)
    width = wrap_width if wrap_width is not None else _auto_wrap_width(fig)
    wrapped = textwrap.fill(text, width=width)

    n_lines = wrapped.count("\n") + 1
    # 0.18 inches per line of text at 9pt + a 0.08 in margin above and below.
    reserved_inches = n_lines * 0.18 + 0.16

    original_w, original_h = fig.get_size_inches()
    new_h = original_h + reserved_inches
    fig.set_size_inches(original_w, new_h)

    bottom_fraction = reserved_inches / new_h
    # Anchor existing axes inside the original-height region.
    fig.subplots_adjust(bottom=bottom_fraction + (1 - bottom_fraction) * fig.subplotpars.bottom)

    fig.text(
        0.5,
        bottom_fraction / 2.0,
        wrapped,
        ha="center",
        va="center",
        fontsize=9,
        fontfamily="DejaVu Sans",
        color="#333333",
        style="italic",
    )

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
