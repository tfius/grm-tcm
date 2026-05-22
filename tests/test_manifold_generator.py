import json

import numpy as np
import pandas as pd

from grm_tcm_manifold_generator import ManifoldGeneratorConfig, SyntheticGRMTCMManifoldGenerator


def test_manifold_generator_outputs_trainer_compatible_geometry(tmp_path):
    out = tmp_path / "manifold"
    cfg = ManifoldGeneratorConfig(
        n_subjects=6,
        n_days=12,
        n_lbo_modes=8,
        output_dir=str(out),
        random_seed=7,
        intervention_event_rate=0.0,
    )
    SyntheticGRMTCMManifoldGenerator(cfg).run()

    visits = pd.read_csv(out / "visits.csv")
    latent = pd.read_csv(out / "latent_states.csv")
    metadata = json.load(open(out / "metadata.json"))
    lbo_csv = pd.read_csv(out / "true_lbo_modes.csv")
    lbo_npz = np.load(out / "true_lbo_eigenmodes.npz", allow_pickle=True)

    required = {
        "subject_id", "day", "sleep_quality", "hrv", "next_day_score",
        "flare_next_day", "true_regime_id", "theta_1", "theta_2",
        "theta1_sin", "theta1_cos", "theta2_sin", "theta2_cos",
        "true_lbo_mode_01", "true_lbo_mode_08",
    }
    assert required.issubset(visits.columns)
    assert len(visits) == 72
    assert len(latent) == 72
    assert len(lbo_csv) == 72
    assert lbo_npz["eigenfunctions"].shape == (72, 8)
    assert lbo_npz["eigenvalues"].shape == (8,)
    assert metadata["config"]["n_subjects"] == 6
    assert metadata["config"]["n_days"] == 12
    assert metadata["ground_truth_files"]["true_lbo_modes.csv"]

    last = visits[visits["day"] == visits["day"].max()]
    nonlast = visits[visits["day"] < visits["day"].max()]
    assert last[["next_day_score", "flare_next_day", "crash_next_day"]].isna().all().all()
    assert nonlast[["next_day_score", "flare_next_day", "crash_next_day"]].notna().all().all()
