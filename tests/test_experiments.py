import pandas as pd

from grm_tcm_experiments import ExperimentConfig, run_manifold_suite


def test_manifold_experiment_matrix_smoke(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "experiments"
    cfg = ExperimentConfig(
        data_dir=data_dir,
        output_dir=out_dir,
        graph_modes="feature_only_diffusion",
        generate=True,
        n_subjects=8,
        n_days=14,
        n_lbo_modes=8,
        n_modes=3,
        n_neighbors=5,
        seed=11,
    )

    leaderboard = run_manifold_suite(cfg)

    assert len(leaderboard) == 1
    assert leaderboard.loc[0, "graph_mode"] == "feature_only_diffusion"
    assert pd.notna(leaderboard.loc[0, "next_day_grm_plus_lag_r2"])
    assert pd.notna(leaderboard.loc[0, "saved_grm_lbo_largest_subspace_mean_cos2"])
    assert (out_dir / "manifold_graph_mode_leaderboard.csv").exists()
    assert (out_dir / "manifold_feature_only_diffusion_lbo" / "manifold_lbo_alignment_metrics.json").exists()
