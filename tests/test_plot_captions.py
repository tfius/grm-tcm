"""Caption helper unit tests + registry-coverage test."""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from grm_tcm_plot_captions import CAPTIONS, save_with_caption  # noqa: E402


def _basic_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(5), np.arange(5))
    return fig


def test_save_with_caption_extends_image_height(tmp_path: Path):
    """Captioned image should be taller than the bare figsize at the same dpi."""

    figsize = (8, 4)
    dpi = 100
    bare_height = int(figsize[1] * dpi)
    fig = _basic_fig(figsize=figsize)
    out = tmp_path / "x.png"
    save_with_caption(fig, out, caption="hello world", dpi=dpi)
    assert out.exists() and out.stat().st_size > 0
    width, height = Image.open(out).size
    assert width == int(figsize[0] * dpi)
    assert height > bare_height, f"expected height > {bare_height}, got {height}"


def test_save_with_caption_uses_registry_when_caption_omitted(tmp_path: Path):
    """A known stem should resolve through CAPTIONS without warning."""

    stem = next(iter(CAPTIONS))
    fig = _basic_fig()
    out = tmp_path / f"{stem}.png"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        save_with_caption(fig, out)
        assert not any("No caption registered" in str(x.message) for x in w)
    assert out.exists()


def test_save_with_caption_warns_on_unknown_stem(tmp_path: Path, recwarn):
    fig = _basic_fig()
    out = tmp_path / "definitely_not_in_registry_xyz.png"
    save_with_caption(fig, out)
    assert any("No caption registered" in str(w.message) for w in recwarn.list)


def test_save_with_caption_closes_figure(tmp_path: Path):
    fig = _basic_fig()
    fignum = fig.number
    save_with_caption(fig, tmp_path / "x.png", caption="bye")
    assert fignum not in plt.get_fignums()


# ---------------------------------------------------------------------------
# Registry coverage: every plot stem produced by the pipelines must be
# registered. Hand-maintained list, deliberately. AST parsing of f-string
# stems would be fragile.
# ---------------------------------------------------------------------------


EXPECTED_STEMS = {
    # grm_tcm_diagnostics.py
    "grm_latent_correlation_heatmap",
    "predicted_vs_actual_next_day_score",
    "residual_histogram",
    "grm_modes_scatter_hidden_subtype",
    "grm_modes_scatter_true_regime",
    "grm_modes_scatter_tcm_like_label",
    "mean_grm_modes_by_hidden_subtype",
    "mean_grm_modes_by_true_regime",
    "mean_grm_modes_by_tcm_like_label",
    "true_regime_occupancy_by_hidden_subtype",
    "true_regime_distribution_by_tcm_label",
    "manifold_scatter_3d",
    "graph_eigen_spectrum",
    # grm_tcm_dynamic_grm.py
    "state_source_metric_comparison",
    "rolling_regime_change_score",
    "selected_modes_over_time",
    "selected_modes_saturation",
    "cumulative_spectral_energy",
    "mean_cumulative_spectral_energy",
    "self_resonance_vs_dysregulation",
    "subject_regime_change_score",
    "subject_self_resonance_vs_dysregulation",
    "soft_self_resonance_vs_dysregulation",
    "pooled_transition_reliability",
    "subject_transition_reliability",
    "inferred_state_true_regime_confusion",
    "true_stuck_occupancy_by_hidden_subtype",
    "subject_resonance_vs_true_stuck_occupancy",
    # grm_tcm_dynamic_eval.py
    "transition_log_loss_by_model",
    "transition_reliability",
    "subject_fingerprint_macro_f1",
    "aliased_visits_scatter",
    "aliased_mode_pair_grid",
    "aliased_per_mode_histograms",
    "aliased_nn_entropy_heatmap",
    "verdicts_forest",
    "ablation_attractor_auc",
    # Phase 1 headline plots
    "transductive_vs_inductive_metrics",
    "flare_calibration_raw_vs_temperature",
    "per_subject_performance",
}


def test_every_expected_stem_has_caption():
    missing = EXPECTED_STEMS - set(CAPTIONS)
    assert not missing, f"Missing captions for: {sorted(missing)}"


@pytest.mark.parametrize("stem", sorted(EXPECTED_STEMS))
def test_captions_are_non_empty(stem):
    text = CAPTIONS[stem]
    assert isinstance(text, str) and len(text.strip()) >= 30, f"Caption for {stem} is too short"
