"""Unit tests for the verdicts-table renderer in grm_tcm_dynamic_eval."""

from grm_tcm_dynamic_eval import VERDICT_LABELS, render_verdicts_table


_FIXTURE = {
    "grm_separates_aliased_states": {"magnitude": 0.182, "ci_low": 0.163, "ci_high": 0.201, "passes": True},
    "grm_attractor_auc_lift_aliased": {"magnitude": 0.052, "ci_low": 0.030, "ci_high": 0.075, "passes": True},
    "grm_silhouette_lift_aliased": {"magnitude": -0.026, "ci_low": -0.068, "ci_high": 0.015, "passes": None},
    "grm_beats_strong_baseline_log_loss": {"magnitude": -0.119, "ci_low": -0.164, "ci_high": -0.069, "passes": False},
}


def test_empty_input_returns_empty_string():
    assert render_verdicts_table({}) == ""


def test_table_contains_each_label():
    table = render_verdicts_table(_FIXTURE)
    for key in _FIXTURE:
        assert VERDICT_LABELS[key] in table


def test_table_contains_delta_and_ci_formats():
    table = render_verdicts_table(_FIXTURE)
    assert "+0.182" in table
    assert "[+0.163, +0.201]" in table
    assert "-0.119" in table
    assert "[-0.164, -0.069]" in table


def test_verdict_strings_present():
    table = render_verdicts_table(_FIXTURE)
    assert "PASS" in table
    assert "FAIL" in table
    assert "MARGINAL" in table


def test_box_drawing_chars_present():
    table = render_verdicts_table(_FIXTURE)
    # corners and tees
    for ch in ["┌", "┐", "└", "┘", "├", "┤", "┬", "┴", "┼", "│", "─"]:
        assert ch in table, f"missing box-drawing char {ch!r}"


def test_rows_render_in_label_map_order():
    """Verdicts present in the certificate render in VERDICT_LABELS order, not insertion order."""

    # Insertion order in the fixture deliberately scrambles the canonical order.
    scrambled = {
        "grm_beats_strong_baseline_log_loss": _FIXTURE["grm_beats_strong_baseline_log_loss"],
        "grm_separates_aliased_states": _FIXTURE["grm_separates_aliased_states"],
        "grm_silhouette_lift_aliased": _FIXTURE["grm_silhouette_lift_aliased"],
        "grm_attractor_auc_lift_aliased": _FIXTURE["grm_attractor_auc_lift_aliased"],
    }
    table = render_verdicts_table(scrambled)
    pos_t1 = table.index(VERDICT_LABELS["grm_separates_aliased_states"])
    pos_t2 = table.index(VERDICT_LABELS["grm_attractor_auc_lift_aliased"])
    pos_t4 = table.index(VERDICT_LABELS["grm_silhouette_lift_aliased"])
    pos_ll = table.index(VERDICT_LABELS["grm_beats_strong_baseline_log_loss"])
    assert pos_t1 < pos_t2 < pos_t4 < pos_ll, "rows should follow VERDICT_LABELS order, not insertion order"


def test_unknown_keys_fall_back_to_raw_key():
    extra = {"mystery_verdict": {"magnitude": 0.5, "ci_low": 0.1, "ci_high": 0.9, "passes": True}}
    table = render_verdicts_table({**_FIXTURE, **extra})
    assert "mystery_verdict" in table


def test_missing_ci_renders_empty_cell():
    no_ci = {"grm_separates_aliased_states": {"magnitude": 0.18, "ci_low": None, "ci_high": None, "passes": True}}
    table = render_verdicts_table(no_ci)
    # Should still render without crashing and include the label + delta.
    assert VERDICT_LABELS["grm_separates_aliased_states"] in table
    assert "+0.180" in table
