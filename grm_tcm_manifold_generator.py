from __future__ import annotations

"""
Continuous-manifold synthetic GRM-TCM generator (v3 improvement #3).

Motivation:
  The existing generator (grm_tcm_synthetic_generator.py) is a discrete
  regime-switching state-space. That construction inherently favors graph-based
  spectral methods because the regimes ARE the discrete clusters. Pang et al.
  2023 (Nature, "Geometric constraints on human brain function") show that
  brain dynamics are better explained by GEOMETRIC eigenmodes of the cortical
  surface than by graph eigenmodes of the connectome. This generator builds a
  body-state analogue: trajectories on a CONTINUOUS low-dim manifold (2D torus),
  with observations as linear projections from manifold coordinates.

Why a torus:
  - Closed surface (no boundary handling).
  - Analytical Laplace-Beltrami eigenmodes (Fourier basis: exp(i(n*theta_1 + m*theta_2))).
  - Topologically nontrivial enough to break naive Euclidean methods.

What it tests:
  - Does GRM's graph-Laplacian eigenbasis approximate the true LBO eigenbasis?
  - Does sampling non-uniformly on the manifold (basins of attraction) bias the
    graph-Laplacian convergence to the LBO?
  - Does per-subject curvature (basin landscape) become recoverable from
    trajectory shape (not just mean position)?

Output schema:
  synthetic_grm_tcm_manifold/
    subjects.csv          - subject_id, hidden_subtype, constitution_*, basin_offset_*
    visits.csv            - subject_id, day, observations, theta_1, theta_2, basin_id,
                            true_regime_id, next_day_score, flare_next_day, ...
    latent_states.csv     - theta_1, theta_2 per visit (the manifold coordinates)
    true_lbo_eigenmodes.npz - eigenvalues, eigenfunctions of LBO on torus
    metadata.json         - schema, design notes

Compatible enough with the trainer that grm_tcm_train.py can consume it directly
(same target columns, observation column names, latent column expectations).

Run:
  python grm_tcm_manifold_generator.py
  python grm_tcm_manifold_generator.py --difficulty hard --seed 1
  python grm_tcm_manifold_generator.py --output-dir synthetic_grm_tcm_manifold
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Observation channel names: identical to the regime-switching generator so the
# trainer can consume the output unchanged.
OBSERVATION_NAMES: List[str] = [
    "sleep_quality", "hrv", "resting_hr", "body_temp",
    "fatigue", "pain", "appetite", "bowel_quality",
    "mood_calm", "energy", "heaviness", "cold_hot",
]

# Manifold dimension. We use a 2D flat torus T^2 = [0, 2π) × [0, 2π).
MANIFOLD_DIM = 2
TORUS_PERIOD = 2.0 * np.pi

# Basin attractors on the torus. Each basin maps to a discrete "regime" label
# so downstream code that expects regime labels still gets meaningful values.
# Positions chosen to be roughly uniform on the torus.
BASIN_NAMES: List[str] = [
    "balanced",           # 0: low-severity attractor
    "stressed",           # 1
    "inflammatory",       # 2
    "digestive",          # 3
    "stuck_depleted",     # 4 (deep + sticky)
    "stuck_agitated",     # 5 (deep + sticky)
    "recovery",           # 6
]
N_BASINS = len(BASIN_NAMES)
STUCK_BASIN_IDS: List[int] = [4, 5]

# Basin centers in (theta_1, theta_2) space. Spread roughly evenly on the torus.
BASIN_CENTERS = np.array([
    [0.0 * np.pi, 0.0 * np.pi],   # 0 balanced: origin
    [0.5 * np.pi, 1.0 * np.pi],   # 1 stressed
    [1.0 * np.pi, 0.5 * np.pi],   # 2 inflammatory
    [1.5 * np.pi, 1.5 * np.pi],   # 3 digestive
    [1.5 * np.pi, 0.5 * np.pi],   # 4 stuck_depleted
    [0.5 * np.pi, 1.5 * np.pi],   # 5 stuck_agitated
    [1.0 * np.pi, 1.5 * np.pi],   # 6 recovery
])

# Basin depth (negative potential well minimum). Deeper basins are stickier.
BASIN_DEPTH = np.array([0.40, 0.55, 0.55, 0.55, 0.80, 0.80, 0.45])

# Basin width (gaussian sigma). Narrower basins are more focal.
BASIN_WIDTH = np.array([0.85, 0.70, 0.70, 0.70, 0.55, 0.55, 0.75])

# Observation projection: x[i] = C_sin @ sin(theta) + C_cos @ cos(theta) + offset + noise.
# Using sin/cos features keeps observations periodic (consistent with torus topology).
# Each row gives (sin_theta1, sin_theta2, cos_theta1, cos_theta2) weights for one channel.
C_TRIG = np.array([
    # sin1   sin2   cos1   cos2
    [-0.25, -0.15, +0.40, +0.20],  # sleep_quality  (positive when near origin)
    [-0.20, -0.30, +0.30, +0.10],  # hrv
    [+0.30, +0.10, -0.20, -0.05],  # resting_hr
    [+0.20, +0.30, +0.10, -0.10],  # body_temp
    [+0.35, +0.20, -0.30, -0.20],  # fatigue
    [+0.30, +0.15, -0.15, -0.20],  # pain
    [-0.20, -0.15, +0.30, +0.15],  # appetite
    [-0.10, -0.20, +0.30, +0.20],  # bowel_quality
    [-0.30, -0.20, +0.20, +0.10],  # mood_calm
    [-0.40, -0.20, +0.30, +0.10],  # energy
    [+0.30, +0.10, -0.10, -0.05],  # heaviness
    [+0.20, +0.30, -0.10, +0.10],  # cold_hot
])

OBS_BASELINE: Dict[str, float] = {
    "sleep_quality": 7.0, "hrv": 58.0, "resting_hr": 66.0, "body_temp": 36.7,
    "fatigue": 3.5, "pain": 2.8, "appetite": 7.0, "bowel_quality": 3.9,
    "mood_calm": 6.8, "energy": 6.8, "heaviness": 3.4, "cold_hot": 5.0,
}
OBS_SCALE: Dict[str, float] = {
    "sleep_quality": 1.5, "hrv": 11.0, "resting_hr": 8.0, "body_temp": 0.35,
    "fatigue": 2.2, "pain": 2.0, "appetite": 1.8, "bowel_quality": 1.0,
    "mood_calm": 1.8, "energy": 2.0, "heaviness": 2.0, "cold_hot": 1.5,
}
OBS_RANGES: Dict[str, Tuple[float, float]] = {
    "sleep_quality": (1.0, 10.0), "hrv": (10.0, 120.0),
    "resting_hr": (40.0, 120.0), "body_temp": (35.2, 39.5),
    "fatigue": (0.0, 10.0), "pain": (0.0, 10.0),
    "appetite": (0.0, 10.0), "bowel_quality": (1.0, 5.0),
    "mood_calm": (0.0, 10.0), "energy": (0.0, 10.0),
    "heaviness": (0.0, 10.0), "cold_hot": (0.0, 10.0),
}

# Constitution axes are inherited from the regime generator for compatibility.
CONSTITUTION_NAMES: List[str] = [
    "constitution_thermal", "constitution_energy", "constitution_stability",
]
N_CONSTITUTION = len(CONSTITUTION_NAMES)

# Per-subtype constitution bias (same semantic intent as v2 regime generator).
CONSTITUTION_SUBTYPE_BIAS: Dict[int, np.ndarray] = {
    0: np.array([-0.30, -0.40, +0.10]),
    1: np.array([+0.40, +0.30, -0.40]),
    2: np.array([+0.50, -0.20, -0.10]),
}

# Constitution -> basin-position offset on the torus. Each constitution axis
# shifts the subject's perceived basin locations by a few radians, so different
# subjects effectively experience different attractor landscapes. This is the
# Pang-style "subject-specific curvature" interpretation of constitution.
# Shape: (3 axes, 2 manifold dims). Magnitude ~0.3 rad keeps it within a basin's width.
BASIN_OFFSET_PROJECTION = np.array([
    [+0.40, +0.10],   # thermal: shifts basins along theta_1
    [+0.10, +0.30],   # energy: shifts basins along theta_2
    [+0.00, +0.00],   # stability: modulates depth, not position (see below)
])


@dataclass
class DifficultyProfile:
    """Difficulty-scaled simulator knobs."""

    sde_noise_std: float          # SDE diffusion strength
    obs_noise_std: float          # observation noise (in projected units)
    missing_rate: float
    intervention_event_rate: float
    constitution_strength: float
    basin_attraction_strength: float  # drift gain toward nearest basin
    label_noise_rate: float


DIFFICULTY_PRESETS: Dict[str, DifficultyProfile] = {
    "easy": DifficultyProfile(
        sde_noise_std=0.20, obs_noise_std=0.30, missing_rate=0.01,
        intervention_event_rate=0.10, constitution_strength=0.50,
        basin_attraction_strength=0.40, label_noise_rate=0.00,
    ),
    "medium": DifficultyProfile(
        sde_noise_std=0.35, obs_noise_std=0.45, missing_rate=0.03,
        intervention_event_rate=0.12, constitution_strength=0.40,
        basin_attraction_strength=0.30, label_noise_rate=0.05,
    ),
    "hard": DifficultyProfile(
        sde_noise_std=0.55, obs_noise_std=0.70, missing_rate=0.08,
        intervention_event_rate=0.15, constitution_strength=0.28,
        basin_attraction_strength=0.22, label_noise_rate=0.12,
    ),
    "chaotic": DifficultyProfile(
        sde_noise_std=0.80, obs_noise_std=0.95, missing_rate=0.15,
        intervention_event_rate=0.19, constitution_strength=0.15,
        basin_attraction_strength=0.15, label_noise_rate=0.22,
    ),
}


@dataclass
class ManifoldGeneratorConfig:
    """Manifold-generator configuration."""

    n_subjects: int = 200
    n_days: int = 120
    random_seed: int = 42
    output_dir: str = "synthetic_grm_tcm_manifold"
    difficulty: str = "medium"
    # LBO truncation for the saved analytical eigenbasis.
    n_lbo_modes: int = 24

    sde_noise_std: Optional[float] = None
    obs_noise_std: Optional[float] = None
    missing_rate: Optional[float] = None
    intervention_event_rate: Optional[float] = None
    constitution_strength: Optional[float] = None
    basin_attraction_strength: Optional[float] = None
    label_noise_rate: Optional[float] = None

    flare_threshold_logit: float = 2.2
    crash_threshold_logit: float = 3.6

    def __post_init__(self) -> None:
        if self.difficulty not in DIFFICULTY_PRESETS:
            raise ValueError(f"Unknown difficulty: {self.difficulty}. Choose from {sorted(DIFFICULTY_PRESETS)}")
        for key, value in asdict(DIFFICULTY_PRESETS[self.difficulty]).items():
            if getattr(self, key) is None:
                setattr(self, key, value)


def _wrap(theta: np.ndarray) -> np.ndarray:
    """Wrap angles into [0, 2π)."""
    return np.mod(theta, TORUS_PERIOD)


def _torus_distance(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    """Geodesic distance on the flat torus. Both inputs shape (..., 2)."""
    diff = np.mod(theta_a - theta_b + np.pi, TORUS_PERIOD) - np.pi
    return np.linalg.norm(diff, axis=-1)


def _potential_gradient(
    theta: np.ndarray,
    basin_centers: np.ndarray,
    basin_depths: np.ndarray,
    basin_widths: np.ndarray,
) -> np.ndarray:
    """Compute -∇U(theta), the drift toward the basin landscape.

    Potential is a sum of negative gaussians (basins). The gradient pulls toward
    the deepest nearby basin, with strength scaled by basin depth and inversely
    by width^2.
    """

    # Vector from theta to each basin (with torus-wrapping).
    diff = np.mod(basin_centers - theta + np.pi, TORUS_PERIOD) - np.pi  # (N_basins, 2)
    sq_dist = np.sum(diff ** 2, axis=-1)
    # Gaussian basin contributions
    weights = basin_depths * np.exp(-sq_dist / (2.0 * basin_widths ** 2))
    # -dU/dtheta = sum_b weight_b * diff_b / width_b^2
    grad = np.sum((weights / (basin_widths ** 2))[:, None] * diff, axis=0)
    return grad


def _nearest_basin(
    theta: np.ndarray,
    basin_centers: np.ndarray,
) -> int:
    """Return basin id of the nearest center under torus distance."""
    return int(np.argmin(_torus_distance(theta[None, :], basin_centers)))


def _lbo_eigenmodes(
    n_modes: int,
    n_samples: int,
    thetas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute the first n_modes Laplace-Beltrami eigenmodes of T^2 evaluated at
    n_samples points (thetas of shape (n_samples, 2)).

    On the flat torus, LBO eigenfunctions are products of sines and cosines:
      psi_{m,n}(theta_1, theta_2) = exp(i (m*theta_1 + n*theta_2))
    with eigenvalues lambda_{m,n} = m^2 + n^2.

    We use real sin/cos basis ordered by increasing eigenvalue. Returns
    (eigenvalues, eigenfunctions, mode_names) where eigenfunctions has shape
    (n_samples, n_modes).
    """

    pairs: List[Tuple[int, int, float]] = [(0, 0, 0.0)]
    max_freq = int(np.ceil(np.sqrt(n_modes))) + 1
    for m in range(-max_freq, max_freq + 1):
        for n in range(-max_freq, max_freq + 1):
            if m == 0 and n == 0:
                continue
            pairs.append((m, n, float(m * m + n * n)))
    pairs.sort(key=lambda t: t[2])

    eigenvalues: List[float] = []
    mode_names: List[str] = []
    columns: List[np.ndarray] = []
    seen_pairs = set()
    for m, n, lam in pairs:
        # Skip the negation pairs we've already covered: combine (m,n) and (-m,-n)
        # into one real basis function.
        key = (abs(m), abs(n), 0 if m * n >= 0 else 1)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        # Real basis: cos and sin of (m*theta_1 + n*theta_2)
        arg = m * thetas[:, 0] + n * thetas[:, 1]
        cos_col = np.cos(arg)
        if not np.allclose(cos_col, 0):
            columns.append(cos_col)
            eigenvalues.append(lam)
            mode_names.append(f"cos_{m}_{n}")
            if len(eigenvalues) >= n_modes:
                break
        if (m, n) != (0, 0):
            sin_col = np.sin(arg)
            if not np.allclose(sin_col, 0):
                columns.append(sin_col)
                eigenvalues.append(lam)
                mode_names.append(f"sin_{m}_{n}")
                if len(eigenvalues) >= n_modes:
                    break
    eigvals = np.asarray(eigenvalues[:n_modes], dtype=float)
    eigfns = np.column_stack(columns[:n_modes]).astype(float)
    return eigvals, eigfns, mode_names[:n_modes]


class SyntheticGRMTCMManifoldGenerator:
    """Continuous-manifold synthetic GRM-TCM generator (v3)."""

    def __init__(self, config: ManifoldGeneratorConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.subject_constitution: Dict[int, np.ndarray] = {}
        self.subject_basin_offset: Dict[int, np.ndarray] = {}

    def run(self) -> Dict[str, pd.DataFrame]:
        subjects = self._generate_subjects()
        visits, latent, events = self._simulate(subjects)
        visits = self._add_outcomes(visits, latent)
        visits = self._add_semantic_labels(visits)
        visits = self._inject_missingness(visits)
        self._write_outputs(subjects, visits, latent, events)
        return {
            "subjects": subjects, "visits": visits,
            "latent_states": latent, "events": events,
        }

    def _generate_subjects(self) -> pd.DataFrame:
        rows: List[Dict] = []
        for sid in range(self.cfg.n_subjects):
            subtype = int(self.rng.integers(0, 3))
            constitution = self.rng.normal(0.0, 1.0, N_CONSTITUTION)
            constitution += CONSTITUTION_SUBTYPE_BIAS[subtype] * self.cfg.constitution_strength
            self.subject_constitution[sid] = constitution
            # Per-subject basin offset: constitution rotates/translates the basin
            # landscape on the torus. Stability constitution modulates basin depth
            # (handled in simulate).
            basin_offset = (constitution[:N_CONSTITUTION - 1] @ BASIN_OFFSET_PROJECTION[:N_CONSTITUTION - 1]).copy()
            self.subject_basin_offset[sid] = basin_offset
            rows.append({
                "subject_id": sid,
                "hidden_subtype": subtype,
                **{name: float(constitution[i]) for i, name in enumerate(CONSTITUTION_NAMES)},
                "basin_offset_theta1": float(basin_offset[0]),
                "basin_offset_theta2": float(basin_offset[1]),
                "sensitivity_intervention": float(self.rng.uniform(0.7, 1.4)),
            })
        return pd.DataFrame(rows)

    def _simulate(self, subjects: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        visit_rows: List[Dict] = []
        latent_rows: List[Dict] = []
        event_rows: List[Dict] = []

        for subj in subjects.itertuples(index=False):
            sid = int(subj.subject_id)
            constitution = self.subject_constitution[sid]
            # Subject-specific basin landscape: shift centers by subject offset,
            # modulate depths by stability (more stable -> deeper balanced+recovery,
            # shallower stuck states).
            subj_centers = _wrap(BASIN_CENTERS + self.subject_basin_offset[sid][None, :])
            stability_mod = float(np.clip(0.7 + 0.3 * constitution[2], 0.3, 1.5))
            subj_depths = BASIN_DEPTH.copy()
            subj_depths[[0, 6]] *= stability_mod          # balanced/recovery deeper if stable
            subj_depths[[4, 5]] /= max(stability_mod, 0.3)  # stuck shallower if stable

            # Initial theta: random near balanced basin.
            theta = _wrap(subj_centers[0] + self.rng.normal(0.0, 0.40, MANIFOLD_DIM))
            prev_basin_id: Optional[int] = None
            dwell_time = 0

            for day in range(self.cfg.n_days):
                # Intervention event: random with subject sensitivity.
                events_today: List[str] = []
                if self.rng.random() < self.cfg.intervention_event_rate:
                    events_today.append("intervention_event")
                    event_rows.append({"subject_id": sid, "day": day, "event_type": "intervention_event"})

                # SDE step: theta_{t+1} = theta_t + drift * dt + noise * sqrt(dt).
                # dt = 1.0 (one day per step), so the SDE is discrete-time but
                # mathematically equivalent to a Langevin step.
                drift = self.cfg.basin_attraction_strength * _potential_gradient(
                    theta, subj_centers, subj_depths, BASIN_WIDTH,
                )
                if events_today:
                    # Intervention nudges toward balanced/recovery basins.
                    nudge = subj_centers[6] - theta  # toward recovery basin
                    nudge = np.mod(nudge + np.pi, TORUS_PERIOD) - np.pi
                    drift = drift + 0.50 * float(subj.sensitivity_intervention) * nudge

                noise = self.rng.normal(0.0, self.cfg.sde_noise_std, MANIFOLD_DIM)
                theta_next = _wrap(theta + drift + noise)

                # Nearest basin as a discrete regime label (for backwards-compatible columns).
                basin_id = _nearest_basin(theta_next, subj_centers)
                if prev_basin_id is None or basin_id != prev_basin_id:
                    dwell_time = 1
                else:
                    dwell_time += 1
                prev_basin_id = basin_id
                is_stuck = int(basin_id in STUCK_BASIN_IDS)

                # Project to observations.
                obs, raw = self._sample_observations(theta_next)

                visit_rows.append({
                    "subject_id": sid,
                    "day": day,
                    "theta_1": float(theta_next[0]),
                    "theta_2": float(theta_next[1]),
                    "basin_id": basin_id,
                    "basin_name": BASIN_NAMES[basin_id],
                    "theta1_sin": float(np.sin(theta_next[0])),
                    "theta1_cos": float(np.cos(theta_next[0])),
                    "theta2_sin": float(np.sin(theta_next[1])),
                    "theta2_cos": float(np.cos(theta_next[1])),
                    # Compatibility columns for downstream code that reads regime labels.
                    "true_regime_id": basin_id,
                    "true_regime": BASIN_NAMES[basin_id],
                    "attractor_state": is_stuck,
                    "dwell_time": int(dwell_time),
                    "latent_instability": float(np.linalg.norm(noise)),
                    "delayed_stress_load": 0.0,
                    "delayed_treatment_load": float(len(events_today) > 0),
                    "delayed_recovery_load": 0.0,
                    "subject_transition_bias_id": sid,
                    **obs,
                })
                latent_rows.append({
                    "subject_id": sid, "day": day,
                    # Standard latent column names that the trainer expects.
                    # We expose theta as the manifold's intrinsic latent.
                    "vitality_depletion": float(np.sin(theta_next[0])),
                    "stress_activation": float(np.sin(theta_next[1])),
                    "inflammatory_load": float(np.cos(theta_next[0])),
                    "digestive_instability": float(np.cos(theta_next[1])),
                    "theta_1": float(theta_next[0]),
                    "theta_2": float(theta_next[1]),
                })
                theta = theta_next

        return pd.DataFrame(visit_rows), pd.DataFrame(latent_rows), pd.DataFrame(event_rows)

    def _sample_observations(self, theta: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """Project theta to observations via sin/cos features + noise."""
        trig = np.array([np.sin(theta[0]), np.sin(theta[1]), np.cos(theta[0]), np.cos(theta[1])])
        raw = C_TRIG @ trig + self.rng.normal(0.0, self.cfg.obs_noise_std, len(OBSERVATION_NAMES))
        out: Dict[str, float] = {}
        for i, name in enumerate(OBSERVATION_NAMES):
            val = OBS_BASELINE[name] + OBS_SCALE[name] * raw[i]
            lo, hi = OBS_RANGES[name]
            out[name] = float(np.clip(val, lo, hi))
        return out, raw

    def _add_outcomes(self, visits: pd.DataFrame, latent: pd.DataFrame) -> pd.DataFrame:
        """Compute global severity and next-day prediction targets."""
        df = visits.sort_values(["subject_id", "day"]).reset_index(drop=True).copy()

        # Global score: linear-in-observations severity.
        df["global_dysregulation_score"] = ((
            0.20 * df["fatigue"]
            + 0.17 * df["pain"]
            + 0.18 * (10.0 - df["sleep_quality"])
            + 0.10 * (10.0 - df["mood_calm"])
            + 0.10 * (10.0 - df["energy"])
            + 0.10 * df["heaviness"]
            + 0.08 * (10.0 - df["appetite"])
            + 0.07 * (5.0 - df["bowel_quality"]) * 2.0
        ) / 10.0).clip(0.0, 1.0)

        # Next-day outcomes depend on tomorrow's basin and stickiness.
        next_basin = df.groupby("subject_id")["basin_id"].shift(-1)
        next_stuck = next_basin.isin(STUCK_BASIN_IDS).astype(float)
        last_day_mask = next_basin.isna()

        raw_score = (
            0.40 * df["global_dysregulation_score"]
            + 0.35 * next_stuck
            + 0.10 * df["latent_instability"]
            + 0.05 * self.rng.normal(0.0, 1.0, len(df))
        )
        df["next_day_score"] = np.clip(raw_score, 0.0, 1.5)
        df.loc[last_day_mask, "next_day_score"] = np.nan

        flare_logit = (
            1.4 * next_stuck
            + 0.30 * (df["global_dysregulation_score"] - 0.5)
            + 0.20 * self.rng.normal(0.0, 1.0, len(df))
        )
        flare_prob = 1.0 / (1.0 + np.exp(-(flare_logit - self.cfg.flare_threshold_logit)))
        crash_prob = 1.0 / (1.0 + np.exp(-(flare_logit - self.cfg.crash_threshold_logit)))
        draws = self.rng.random(len(df))
        df["flare_next_day"] = pd.array(np.where(draws < flare_prob, 1, 0), dtype="Int64")
        df["crash_next_day"] = pd.array(np.where(draws < crash_prob, 1, 0), dtype="Int64")
        df.loc[last_day_mask, ["flare_next_day", "crash_next_day"]] = pd.NA

        score_p2 = df.groupby("subject_id")["global_dysregulation_score"].shift(-2)
        worsening_diff = score_p2 - df["global_dysregulation_score"]
        worsening_int = np.where(worsening_diff >= 0.18, 1, 0)
        df["worsening_2day"] = pd.array(
            np.where(worsening_diff.isna(), pd.NA, worsening_int), dtype="Int64",
        )
        return df

    def _add_semantic_labels(self, visits: pd.DataFrame) -> pd.DataFrame:
        """Coarse TCM-like labels keyed off basin identity + obs noise."""
        df = visits.copy()
        basin_label_map = {
            0: "balanced_flow",
            1: "stuck_agitated",
            2: "hot_overactive",
            3: "heavy_damp_like",
            4: "depleted",
            5: "stuck_agitated",
            6: "balanced_flow",
        }
        tcm_label_map = {
            0: "mixed_pattern_like",
            1: "liver_qi_stagnation_like",
            2: "damp_heat_like",
            3: "spleen_qi_deficiency_like",
            4: "spleen_qi_deficiency_like",
            5: "liver_qi_stagnation_like",
            6: "mixed_pattern_like",
        }
        df["qi_like_label"] = df["basin_id"].map(basin_label_map)
        df["tcm_like_label"] = df["basin_id"].map(tcm_label_map)
        df["contrarian_signature"] = "canonical_single_cluster"

        rate = float(np.clip(self.cfg.label_noise_rate, 0.0, 1.0))
        if rate > 0.0:
            qi_choices = np.array(sorted(df["qi_like_label"].dropna().unique()))
            tcm_choices = np.array(sorted(df["tcm_like_label"].dropna().unique()))
            mask = self.rng.random(len(df)) < rate
            if len(qi_choices):
                df.loc[mask, "qi_like_label"] = self.rng.choice(qi_choices, size=int(mask.sum()))
            if len(tcm_choices):
                df.loc[mask, "tcm_like_label"] = self.rng.choice(tcm_choices, size=int(mask.sum()))
            df.loc[mask, "contrarian_signature"] = "label_noise_injected"
        return df

    def _inject_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        rate = float(self.cfg.missing_rate)
        for col in OBSERVATION_NAMES:
            mask = self.rng.random(len(out)) < rate
            out.loc[mask, col] = np.nan
        return out

    def _write_outputs(
        self,
        subjects: pd.DataFrame,
        visits: pd.DataFrame,
        latent: pd.DataFrame,
        events: pd.DataFrame,
    ) -> None:
        # Compute and save analytical LBO eigenmodes at the visit thetas. These
        # are the GROUND TRUTH eigenmodes that GRM's graph-Laplacian eigenvectors
        # should approximate if the graph captures the manifold geometry.
        thetas = visits[["theta_1", "theta_2"]].to_numpy(float)
        lbo_eigvals, lbo_eigfns, lbo_mode_names = _lbo_eigenmodes(self.cfg.n_lbo_modes, len(thetas), thetas)
        lbo_mode_cols = [f"true_lbo_mode_{idx:02d}" for idx in range(1, lbo_eigfns.shape[1] + 1)]
        lbo_df = visits[["subject_id", "day"]].copy()
        for idx, col in enumerate(lbo_mode_cols):
            visits[col] = lbo_eigfns[:, idx]
            latent[col] = lbo_eigfns[:, idx]
            lbo_df[col] = lbo_eigfns[:, idx]

        events_out = events
        if events_out.empty:
            events_out = pd.DataFrame(columns=["subject_id", "day", "event_type"])

        subjects.to_csv(self.output_dir / "subjects.csv", index=False)
        visits.to_csv(self.output_dir / "visits.csv", index=False)
        latent.to_csv(self.output_dir / "latent_states.csv", index=False)
        events_out.to_csv(self.output_dir / "events.csv", index=False)
        lbo_df.to_csv(self.output_dir / "true_lbo_modes.csv", index=False)
        np.savez_compressed(
            self.output_dir / "true_lbo_eigenmodes.npz",
            eigenvalues=lbo_eigvals,
            eigenfunctions=lbo_eigfns,
            mode_names=np.asarray(lbo_mode_names, dtype=object),
            mode_columns=np.asarray(lbo_mode_cols, dtype=object),
            visit_ids=np.arange(len(thetas)),
            subject_ids=visits["subject_id"].to_numpy(int),
            days=visits["day"].to_numpy(int),
        )

        metadata = {
            "generator_kind": "continuous_manifold",
            "config": asdict(self.cfg),
            "manifold": "T^2 (flat 2-torus, period 2π)",
            "observation_names": OBSERVATION_NAMES,
            "constitution_names": CONSTITUTION_NAMES,
            "basin_names": BASIN_NAMES,
            "stuck_basin_ids": STUCK_BASIN_IDS,
            "lbo_mode_columns": lbo_mode_cols,
            "lbo_mode_names": lbo_mode_names,
            "lbo_eigenvalues": [float(x) for x in lbo_eigvals],
            "ground_truth_files": {
                "subjects.csv": "Subject constitution and subject-specific basin offsets.",
                "visits.csv": "Trainer-compatible visits plus theta coordinates, trig coordinates, and true_lbo_mode_* columns.",
                "latent_states.csv": "Intrinsic manifold coordinates and true LBO mode values per subject/day.",
                "events.csv": "Intervention event log.",
                "true_lbo_modes.csv": "Per-visit true LBO mode table keyed by subject_id/day.",
                "true_lbo_eigenmodes.npz": "Analytical torus eigenvalues/eigenfunctions evaluated at every visit.",
            },
            "design_notes": {
                "framing": (
                    "Continuous-manifold variant (v3 #3, Pang-inspired). Trajectories live on "
                    "a 2D torus; observations are sin/cos projections of theta. The true LBO "
                    "eigenmodes (Fourier basis) are saved in true_lbo_eigenmodes.npz for direct "
                    "comparison against GRM's graph-Laplacian eigenvectors."
                ),
                "basins": (
                    "Seven basin attractors map to discrete regime labels for compatibility with "
                    "the regime-switching pipeline. Subject-specific basin offsets and depth "
                    "modulation by constitution_stability give each subject a slightly different "
                    "landscape — the manifold equivalent of constitution-dynamics coupling."
                ),
                "limitations": (
                    "Not a biological simulator. Tests whether graph-Laplacian eigenvectors "
                    "converge to LBO eigenfunctions under non-uniform manifold sampling."
                ),
            },
            "framing": (
                "Continuous-manifold synthetic benchmark. Designed to test the Pang-style "
                "geometry-vs-connectivity question on a body-state substrate."
            ),
            "recommended_tasks": [
                "Compare GRM graph-Laplacian eigenvectors against analytical LBO eigenmodes.",
                "Recover constitution (basin offset) from per-subject trajectory shape.",
                "Predict next_day_score / flare_next_day using only continuous theta coords.",
                "Test whether non-uniform sampling biases graph-Laplacian convergence to LBO.",
            ],
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)


def parse_args() -> ManifoldGeneratorConfig:
    parser = argparse.ArgumentParser(description="Generate the continuous-manifold synthetic GRM-TCM dataset.")
    parser.add_argument("--output-dir", default="synthetic_grm_tcm_manifold")
    parser.add_argument("--difficulty", default="medium", choices=sorted(DIFFICULTY_PRESETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-subjects", type=int, default=ManifoldGeneratorConfig.n_subjects)
    parser.add_argument("--n-days", type=int, default=ManifoldGeneratorConfig.n_days)
    parser.add_argument("--n-lbo-modes", type=int, default=ManifoldGeneratorConfig.n_lbo_modes)
    args = parser.parse_args()
    return ManifoldGeneratorConfig(
        n_subjects=args.n_subjects,
        n_days=args.n_days,
        random_seed=args.seed,
        output_dir=args.output_dir,
        difficulty=args.difficulty,
        n_lbo_modes=args.n_lbo_modes,
    )


if __name__ == "__main__":
    cfg = parse_args()
    gen = SyntheticGRMTCMManifoldGenerator(cfg)
    outputs = gen.run()
    print(f"Wrote manifold synthetic dataset to: {gen.output_dir.resolve()}")
    print("Files: subjects.csv, visits.csv, latent_states.csv, events.csv, "
          "true_lbo_modes.csv, true_lbo_eigenmodes.npz, metadata.json")
    v = outputs["visits"]
    print(f"\nBasin occupancy:")
    print(v["basin_name"].value_counts(normalize=True).round(3).to_string())
    print(f"\nFlare rate: {float(v['flare_next_day'].dropna().astype(int).mean()):.3f}")
    print(f"Theta coverage: theta_1 in [{v['theta_1'].min():.2f}, {v['theta_1'].max():.2f}], "
          f"theta_2 in [{v['theta_2'].min():.2f}, {v['theta_2'].max():.2f}]")
