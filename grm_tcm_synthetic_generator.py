from __future__ import annotations

"""
GRM-TCM synthetic dataset generator (regime-switching redesign).

Creates a controlled longitudinal benchmark with:
- discrete latent regimes (balanced, stressed, inflammatory, digestive, stuck_*, recovery)
- continuous latent state z_t evolving under regime-conditional dynamics
- subject-specific transition bias matrices driven by hidden_subtype
- stuck attractor regimes with high self-transition probability
- delayed stress / treatment / recovery loads
- regime-conditional observation aliasing (different regimes produce similar x)
- outcomes that depend on tomorrow's regime, dwell time, and latent instability
  (not just today's observed severity)
- noisy practitioner-style TCM-like labels with intentional ontology mismatch

Framing: synthetic benchmark for latent-state recovery and ontology mismatch
detection. Not a biological simulator. Not evidence for TCM or Qi.

Run:
  python grm_tcm_synthetic_generator.py
  python grm_tcm_synthetic_generator.py --difficulty hard --seed 1

Outputs:
  synthetic_grm_tcm/subjects.csv
  synthetic_grm_tcm/visits.csv
  synthetic_grm_tcm/latent_states.csv
  synthetic_grm_tcm/events.csv
  synthetic_grm_tcm/true_regimes.csv
  synthetic_grm_tcm/true_transition_matrices.csv
  synthetic_grm_tcm/true_attractor_states.csv
  synthetic_grm_tcm/metadata.json
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LATENT_NAMES: List[str] = [
    "vitality_depletion",
    "stress_activation",
    "inflammatory_load",
    "digestive_instability",
]

OBSERVATION_NAMES: List[str] = [
    "sleep_quality", "hrv", "resting_hr", "body_temp",
    "fatigue", "pain", "appetite", "bowel_quality",
    "mood_calm", "energy", "heaviness", "cold_hot",
]

EVENT_TYPES: List[str] = ["stress_event", "recovery_event", "treatment_event"]

REGIME_NAMES: List[str] = [
    "balanced",              # 0
    "stressed_recoverable",  # 1
    "inflammatory",          # 2
    "digestive_instable",    # 3
    "stuck_depleted",        # 4 (sticky attractor)
    "stuck_agitated",        # 5 (sticky attractor)
    "recovery",              # 6 (transient)
]
N_REGIMES = len(REGIME_NAMES)
STUCK_REGIME_IDS: List[int] = [4, 5]
ACTIVE_REGIME_IDS: List[int] = [1, 2, 3]

# Regime-conditional latent prototypes (drift targets) in z-space.
REGIME_PROTOTYPES = np.array([
    [ 0.00,  0.00,  0.00,  0.00],   # balanced
    [ 0.20,  0.85,  0.25,  0.10],   # stressed_recoverable
    [ 0.30,  0.20,  0.90,  0.30],   # inflammatory
    [ 0.20,  0.10,  0.30,  0.85],   # digestive_instable
    [ 0.90,  0.15,  0.45,  0.75],   # stuck_depleted
    [ 0.55,  0.95,  0.70,  0.10],   # stuck_agitated
    [-0.25, -0.25, -0.25, -0.25],   # recovery
], dtype=float)

# Drift strength per regime (how strongly z is pulled toward the prototype).
REGIME_DRIFT_STRENGTH = np.array([0.18, 0.30, 0.32, 0.30, 0.40, 0.40, 0.28])

# Per-regime additional latent noise scale.
REGIME_NOISE_GAIN = np.array([0.90, 1.05, 1.10, 1.05, 1.25, 1.25, 0.85])

# Aliasing groups: which regimes share an observation-offset signature.
# stressed_recoverable and stuck_agitated alias to "wired"; digestive and
# stuck_depleted alias to "heavy"; balanced and recovery alias to "calm".
ALIAS_GROUPS: Dict[int, str] = {
    0: "calm",
    1: "wired",
    2: "hot",
    3: "heavy",
    4: "heavy",
    5: "wired",
    6: "calm",
}

# Per-alias-group observation offsets (in z-projected units, applied after C @ z).
ALIAS_OFFSETS: Dict[str, np.ndarray] = {
    "calm":  np.array([+0.55, +0.55, -0.30,  0.00, -0.95, -0.55, +0.60, +0.10, +0.85, +0.85, -0.85, -0.05]),
    "wired": np.array([-0.85, -0.95, +0.80, +0.05, +0.70, +0.55, -0.30, -0.05, -0.95, -0.65, +0.35, +0.20]),
    "hot":   np.array([-0.40, -0.30, +0.40, +0.65, +0.95, +0.95, -0.30, -0.15, -0.55, -0.55, +0.55, +0.85]),
    "heavy": np.array([-0.30, -0.20,  0.00,  0.00, +0.65, +0.35, -0.95, -0.95, -0.20, -0.65, +0.95, -0.45]),
}


@dataclass
class DifficultyProfile:
    """Difficulty-scaled simulator knobs."""

    latent_noise_std: float
    obs_noise_std: float
    missing_rate: float
    stress_event_rate: float
    recovery_event_rate: float
    treatment_event_rate: float
    hidden_subtype_strength: float
    delayed_treatment_effect: float
    label_noise_rate: float
    practitioner_bias: float
    obs_aliasing_strength: float
    sticky_strength: float
    subject_bias_strength: float
    missing_rate_severe_multiplier: float


DIFFICULTY_PRESETS: Dict[str, DifficultyProfile] = {
    "easy": DifficultyProfile(
        latent_noise_std=0.10, obs_noise_std=0.28, missing_rate=0.01,
        stress_event_rate=0.07, recovery_event_rate=0.09, treatment_event_rate=0.10,
        hidden_subtype_strength=0.70, delayed_treatment_effect=0.10,
        label_noise_rate=0.00, practitioner_bias=0.05,
        obs_aliasing_strength=0.30, sticky_strength=0.05,
        subject_bias_strength=1.40, missing_rate_severe_multiplier=2.0,
    ),
    "medium": DifficultyProfile(
        latent_noise_std=0.18, obs_noise_std=0.45, missing_rate=0.03,
        stress_event_rate=0.10, recovery_event_rate=0.07, treatment_event_rate=0.12,
        hidden_subtype_strength=0.45, delayed_treatment_effect=0.25,
        label_noise_rate=0.05, practitioner_bias=0.20,
        obs_aliasing_strength=0.55, sticky_strength=0.10,
        subject_bias_strength=1.00, missing_rate_severe_multiplier=2.5,
    ),
    "hard": DifficultyProfile(
        latent_noise_std=0.28, obs_noise_std=0.70, missing_rate=0.08,
        stress_event_rate=0.14, recovery_event_rate=0.05, treatment_event_rate=0.15,
        hidden_subtype_strength=0.28, delayed_treatment_effect=0.45,
        label_noise_rate=0.12, practitioner_bias=0.45,
        obs_aliasing_strength=0.80, sticky_strength=0.15,
        subject_bias_strength=0.60, missing_rate_severe_multiplier=3.5,
    ),
    "chaotic": DifficultyProfile(
        latent_noise_std=0.40, obs_noise_std=0.95, missing_rate=0.15,
        stress_event_rate=0.20, recovery_event_rate=0.04, treatment_event_rate=0.19,
        hidden_subtype_strength=0.15, delayed_treatment_effect=0.65,
        label_noise_rate=0.22, practitioner_bias=0.75,
        obs_aliasing_strength=1.10, sticky_strength=0.20,
        subject_bias_strength=0.30, missing_rate_severe_multiplier=5.0,
    ),
}


@dataclass
class GeneratorConfig:
    """Generator configuration. CLI-overridable knobs."""

    n_subjects: int = 80
    n_days: int = 60
    latent_dim: int = 4
    random_seed: int = 42
    output_dir: str = "synthetic_grm_tcm"
    difficulty: str = "medium"

    latent_noise_std: float = 0.18
    obs_noise_std: float = 0.45
    missing_rate: float = 0.03
    stress_event_rate: float = 0.10
    recovery_event_rate: float = 0.07
    treatment_event_rate: float = 0.12
    hidden_subtype_strength: float = 0.45
    delayed_treatment_effect: float = 0.25
    label_noise_rate: float = 0.05
    practitioner_bias: float = 0.20
    obs_aliasing_strength: float = 0.55
    sticky_strength: float = 0.10
    subject_bias_strength: float = 1.00
    missing_rate_severe_multiplier: float = 2.5

    load_decay: float = 0.65
    flare_threshold_logit: float = 2.2
    crash_threshold_logit: float = 3.6

    def __post_init__(self) -> None:
        if self.difficulty not in DIFFICULTY_PRESETS:
            raise ValueError(f"Unknown difficulty: {self.difficulty}. Choose from {sorted(DIFFICULTY_PRESETS)}")
        for key, value in asdict(DIFFICULTY_PRESETS[self.difficulty]).items():
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# Regime model
# ---------------------------------------------------------------------------


def _build_base_transition(sticky_strength: float) -> np.ndarray:
    """Construct the 7x7 base regime transition matrix."""
    base = np.array([
        # bal   str   inf   dig   d_dep d_ag  rec
        [0.78, 0.08, 0.05, 0.05, 0.01, 0.01, 0.02],   # balanced
        [0.14, 0.54, 0.08, 0.04, 0.03, 0.09, 0.08],   # stressed_recoverable
        [0.10, 0.08, 0.54, 0.10, 0.07, 0.03, 0.08],   # inflammatory
        [0.10, 0.04, 0.10, 0.54, 0.09, 0.03, 0.10],   # digestive_instable
        [0.04, 0.04, 0.05, 0.05, 0.62, 0.02, 0.18],   # stuck_depleted
        [0.04, 0.08, 0.04, 0.02, 0.02, 0.62, 0.18],   # stuck_agitated
        [0.42, 0.06, 0.05, 0.05, 0.02, 0.03, 0.37],   # recovery
    ], dtype=float)
    for rid in STUCK_REGIME_IDS:
        boost = float(np.clip(sticky_strength, 0.0, 0.25))
        base[rid] = _shift_diagonal(base[rid], rid, boost)
    return base / base.sum(axis=1, keepdims=True)


def _shift_diagonal(row: np.ndarray, idx: int, amount: float) -> np.ndarray:
    """Add `amount` to row[idx] and subtract proportionally from other entries."""
    other = np.delete(np.arange(len(row)), idx)
    new = row.copy()
    new[idx] += amount
    if new[other].sum() > 0:
        new[other] *= max(1.0 - amount / new[other].sum(), 0.05)
    return new / new.sum()


def _subtype_bias_prototype(subtype: int) -> np.ndarray:
    """Return the 7x7 logit-additive transition bias prototype for a subtype."""
    b = np.zeros((N_REGIMES, N_REGIMES))
    log22 = np.log(2.2)
    log15 = np.log(1.5)
    if subtype == 0:  # digestive responder: drawn into stuck_depleted via dig/inf
        b[2, 4] += +log22
        b[3, 4] += +log22
        b[0, 3] += +log15
        b[4, 6] -= log15
        b[0, 5] -= log15
        b[1, 5] -= log15
    elif subtype == 1:  # stress responder: drawn into stuck_agitated via stress
        b[0, 1] += +log22
        b[1, 5] += +log22
        b[2, 5] += +log15
        b[5, 6] -= log15
        b[0, 4] -= log15
        b[3, 4] -= log15
    else:  # inflammatory responder: drawn into inflammatory then stuck_depleted
        b[0, 2] += +log22
        b[2, 4] += +log15
        b[3, 2] += +log15
        b[4, 6] -= log15
        b[0, 5] -= log15
        b[1, 5] -= log15
    return b


# Event-driven additive logits on regime transitions (independent of c_t).
EVENT_REGIME_PUSH: Dict[str, np.ndarray] = {
    "stress":   np.array([-0.5, +0.7, +0.4, +0.2, +0.2, +0.4, -0.6]),
    "recovery": np.array([+0.6, -0.3, -0.4, -0.3, -0.6, -0.6, +0.8]),
    "treatment": np.array([+0.2, -0.2, -0.3, -0.3, -0.5, -0.5, +0.6]),
}

# Treatment effect on regime transitions, conditional on hidden_subtype.
TREATMENT_PUSH_BY_SUBTYPE: Dict[int, np.ndarray] = {
    0: np.array([+0.3, -0.1, -0.3, -0.4, -0.7, -0.2, +0.7]),
    1: np.array([+0.3, -0.4, -0.2, -0.1, -0.2, -0.7, +0.7]),
    2: np.array([+0.3, -0.2, -0.6, -0.3, -0.6, -0.2, +0.7]),
}

# Stress-event impulse on z (regardless of regime).
STRESS_Z_IMPULSE = np.array([0.06, 0.22, 0.08, 0.04])
RECOVERY_Z_IMPULSE = np.array([0.10, 0.10, 0.08, 0.06])

# Per-subtype treatment z-impulse (mirrors previous behavior).
TREATMENT_Z_PROTOTYPES: Dict[int, np.ndarray] = {
    0: np.array([-0.22, -0.06, -0.10, -0.18]),
    1: np.array([-0.10, -0.22, -0.12, -0.04]),
    2: np.array([-0.08, -0.04, -0.24, -0.10]),
}


# Shared linear backbone (regime drift dominates; this is gentle).
A_SHARED = np.array([
    [0.78, 0.07, 0.05, 0.04],
    [0.05, 0.74, 0.10, 0.03],
    [0.04, 0.06, 0.78, 0.07],
    [0.04, 0.03, 0.07, 0.76],
])

# Observation projection (rows = observed, cols = latent).
C_MATRIX = np.array([
    [-0.35, -0.18, -0.12, -0.10],  # sleep_quality
    [-0.22, -0.38, -0.18, -0.08],  # hrv
    [+0.10, +0.28, +0.14, +0.04],  # resting_hr
    [+0.02, +0.06, +0.25, +0.02],  # body_temp
    [+0.42, +0.18, +0.20, +0.12],  # fatigue
    [+0.15, +0.18, +0.40, +0.08],  # pain
    [-0.10, -0.08, -0.10, -0.34],  # appetite
    [-0.08, -0.04, -0.08, -0.40],  # bowel_quality
    [-0.16, -0.42, -0.10, -0.04],  # mood_calm
    [-0.46, -0.16, -0.14, -0.10],  # energy
    [+0.18, +0.05, +0.16, +0.36],  # heaviness
    [-0.10, +0.08, +0.35, +0.02],  # cold_hot
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


def _softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax."""
    y = x - np.max(x)
    e = np.exp(y)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SyntheticGRMTCMGenerator:
    """Regime-switching synthetic GRM-TCM generator."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transition_matrix = _build_base_transition(config.sticky_strength)
        self.subject_bias_matrices: Dict[int, np.ndarray] = {}

    # ---- top-level orchestration --------------------------------------------

    def run(self) -> Dict[str, pd.DataFrame]:
        """Generate the full dataset and write all outputs."""
        subjects = self._generate_subjects()
        visits, latent, events = self._simulate(subjects)
        visits = self._add_outcomes(visits, latent)
        visits = self._add_semantic_labels(visits, latent)
        visits = self._inject_missingness(visits)
        attractor_summary = self._summarize_attractors(visits)
        self._write_outputs(subjects, visits, latent, events, attractor_summary)
        return {
            "subjects": subjects, "visits": visits,
            "latent_states": latent, "events": events,
            "attractor_summary": attractor_summary,
        }

    # ---- subject generation -------------------------------------------------

    def _generate_subjects(self) -> pd.DataFrame:
        """Generate per-subject metadata including transition-bias matrices."""
        rows: List[Dict] = []
        for sid in range(self.cfg.n_subjects):
            subtype = int(self.rng.integers(0, 3))
            baseline = self.rng.normal(0.0, 0.30, self.cfg.latent_dim)
            # baseline z is a weak prior; the dominant subtype signal is in dynamics
            subtype_offset = REGIME_PROTOTYPES[4 + (subtype % 2)][: self.cfg.latent_dim] * 0.15
            z0 = baseline + subtype_offset * self.cfg.hidden_subtype_strength

            bias_proto = _subtype_bias_prototype(subtype) * self.cfg.subject_bias_strength
            jitter = self.rng.normal(0.0, 0.10 * self.cfg.subject_bias_strength, size=(N_REGIMES, N_REGIMES))
            self.subject_bias_matrices[sid] = bias_proto + jitter

            rows.append({
                "subject_id": sid,
                "hidden_subtype": subtype,
                "baseline_vitality_depletion": float(z0[0]),
                "baseline_stress_activation": float(z0[1]),
                "baseline_inflammatory_load": float(z0[2]),
                "baseline_digestive_instability": float(z0[3]),
                "sensitivity_stress": float(self.rng.uniform(0.7, 1.3)),
                "sensitivity_recovery": float(self.rng.uniform(0.7, 1.2)),
                "sensitivity_treatment": float(self.rng.uniform(0.7, 1.4)),
                "chronic_load": float(self.rng.uniform(0.0, 0.8)),
                "attractor_susceptibility": float(np.clip(self.rng.normal(0.5 + 0.2 * (subtype != 1), 0.20), 0.0, 1.0)),
                "recovery_efficiency": float(np.clip(self.rng.normal(0.55, 0.18), 0.05, 0.95)),
                "subject_transition_bias_id": sid,
            })
        return pd.DataFrame(rows)

    # ---- core simulation ----------------------------------------------------

    def _simulate(self, subjects: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Day-by-day switching state-space simulation per subject."""
        visit_rows: List[Dict] = []
        latent_rows: List[Dict] = []
        event_rows: List[Dict] = []

        for subj in subjects.itertuples(index=False):
            sid = int(subj.subject_id)
            subtype = int(subj.hidden_subtype)
            bias = self.subject_bias_matrices[sid]
            susc = float(subj.attractor_susceptibility)
            recov_eff = float(subj.recovery_efficiency)

            z = np.array([
                subj.baseline_vitality_depletion,
                subj.baseline_stress_activation,
                subj.baseline_inflammatory_load,
                subj.baseline_digestive_instability,
            ], dtype=float)
            c = 0
            dwell = 0
            delayed_stress = 0.0
            delayed_treatment = 0.0
            delayed_recovery = 0.0
            recent_zs: List[np.ndarray] = [z.copy()]

            for day in range(self.cfg.n_days):
                events_today = self._sample_events(z)
                stress_today = float("stress_event" in events_today)
                recovery_today = float("recovery_event" in events_today)
                treatment_today = float("treatment_event" in events_today)

                base_logits = np.log(self.transition_matrix[c] + 1e-9)
                bias_logits = bias[c]
                event_logits = (
                    EVENT_REGIME_PUSH["stress"] * (stress_today + 0.55 * delayed_stress) * float(subj.sensitivity_stress)
                    + EVENT_REGIME_PUSH["recovery"] * (recovery_today + 0.55 * delayed_recovery) * recov_eff
                    + TREATMENT_PUSH_BY_SUBTYPE[subtype] * (treatment_today + 0.55 * delayed_treatment) * float(subj.sensitivity_treatment)
                )
                if c in STUCK_REGIME_IDS:
                    event_logits[c] += susc * 0.35
                    event_logits[6] -= (1.0 - recov_eff) * 0.3

                probs = _softmax(base_logits + bias_logits + event_logits)
                c_next = int(self.rng.choice(N_REGIMES, p=probs))
                dwell = dwell + 1 if c_next == c else 0

                proto = REGIME_PROTOTYPES[c_next]
                alpha = REGIME_DRIFT_STRENGTH[c_next]
                z_next = (1.0 - alpha) * (A_SHARED @ z) + alpha * proto

                if stress_today:
                    z_next += STRESS_Z_IMPULSE * float(subj.sensitivity_stress)
                if recovery_today:
                    z_next -= RECOVERY_Z_IMPULSE * recov_eff
                if treatment_today:
                    z_next += TREATMENT_Z_PROTOTYPES[subtype] * float(subj.sensitivity_treatment)

                noise_scale = self.cfg.latent_noise_std * REGIME_NOISE_GAIN[c_next]
                z_next += self.rng.normal(0.0, noise_scale, self.cfg.latent_dim)

                weekly_rhythm = 0.04 * np.sin(2 * np.pi * day / 7.0)
                z_next[1] += weekly_rhythm

                instability = float(np.linalg.norm(z_next - z))
                if len(recent_zs) >= 2:
                    instability += 0.5 * float(np.linalg.norm(z - recent_zs[-2]))

                obs = self._sample_observations(z_next, c_next)

                # Decay then add today's impulses for next-day forecasting.
                delayed_stress = self.cfg.load_decay * delayed_stress + stress_today
                delayed_recovery = self.cfg.load_decay * delayed_recovery + recovery_today
                delayed_treatment = self.cfg.load_decay * delayed_treatment + treatment_today

                visit_rows.append({
                    "subject_id": sid,
                    "day": day,
                    "true_regime_id": c_next,
                    "true_regime": REGIME_NAMES[c_next],
                    "attractor_state": int(c_next in STUCK_REGIME_IDS),
                    "dwell_time": int(dwell),
                    "latent_instability": float(instability),
                    "delayed_stress_load": float(delayed_stress),
                    "delayed_treatment_load": float(delayed_treatment),
                    "delayed_recovery_load": float(delayed_recovery),
                    "subject_transition_bias_id": sid,
                    **obs,
                })
                latent_rows.append({
                    "subject_id": sid, "day": day,
                    **{name: float(z_next[i]) for i, name in enumerate(LATENT_NAMES)},
                })
                for event_name in events_today:
                    event_rows.append({"subject_id": sid, "day": day, "event_type": event_name})

                recent_zs.append(z_next.copy())
                if len(recent_zs) > 4:
                    recent_zs.pop(0)
                z = z_next
                c = c_next

        return pd.DataFrame(visit_rows), pd.DataFrame(latent_rows), pd.DataFrame(event_rows)

    def _sample_events(self, z: np.ndarray) -> List[str]:
        """Sample today's event set; treatment rate scales with observed severity."""
        events: List[str] = []
        if self.rng.random() < self.cfg.stress_event_rate:
            events.append("stress_event")
        if self.rng.random() < self.cfg.recovery_event_rate:
            events.append("recovery_event")
        dysregulation = float(np.clip(np.mean(np.maximum(z, 0.0)), 0.0, 3.0) / 3.0)
        treatment_rate = self.cfg.treatment_event_rate * (1.0 + self.cfg.practitioner_bias * dysregulation)
        if self.rng.random() < min(treatment_rate, 0.95):
            events.append("treatment_event")
        return events

    def _sample_observations(self, z: np.ndarray, regime_id: int) -> Dict[str, float]:
        """Project z to observations with regime-aliased offsets and noise."""
        alias_key = ALIAS_GROUPS[regime_id]
        alias_vec = ALIAS_OFFSETS[alias_key]
        raw = (
            C_MATRIX @ z
            + self.cfg.obs_aliasing_strength * alias_vec
            + self.rng.normal(0.0, self.cfg.obs_noise_std, len(OBSERVATION_NAMES))
        )
        out: Dict[str, float] = {}
        for i, name in enumerate(OBSERVATION_NAMES):
            val = OBS_BASELINE[name] + OBS_SCALE[name] * raw[i]
            lo, hi = OBS_RANGES[name]
            out[name] = float(np.clip(val, lo, hi))
        return out

    # ---- outcomes -----------------------------------------------------------

    def _add_outcomes(self, visits: pd.DataFrame, latent: pd.DataFrame) -> pd.DataFrame:
        """Compute global severity, next-day outcomes, and probabilistic flares."""
        df = visits.sort_values(["subject_id", "day"]).reset_index(drop=True).copy()

        df["global_dysregulation_score"] = self._global_score(df)

        for shifted in ["true_regime_id", "dwell_time", "latent_instability", "delayed_stress_load", "delayed_recovery_load"]:
            df[f"next_{shifted}"] = df.groupby("subject_id")[shifted].shift(-1)

        next_stuck = df["next_true_regime_id"].isin(STUCK_REGIME_IDS).astype(float)
        next_dwell = df["next_dwell_time"].fillna(0.0)
        next_instability = df["next_latent_instability"].fillna(0.0)
        next_stress = df["next_delayed_stress_load"].fillna(0.0)
        next_recovery = df["next_delayed_recovery_load"].fillna(0.0)

        raw_score = (
            0.30 * df["global_dysregulation_score"]
            + 0.35 * next_stuck * (1.0 + 0.18 * np.tanh(next_dwell / 4.0))
            + 0.20 * np.tanh(next_instability / 1.5)
            + 0.10 * np.tanh(next_stress)
            - 0.05 * np.tanh(next_recovery)
            + 0.05 * self.rng.normal(0.0, 1.0, len(df))
        )
        df["next_day_score"] = np.clip(raw_score, 0.0, 1.5)

        flare_logit = (
            1.4 * next_stuck
            + 0.9 * np.tanh(next_dwell / 4.0)
            + 0.9 * np.tanh(next_instability / 1.5)
            + 0.6 * np.tanh(next_stress)
            - 0.7 * np.tanh(next_recovery)
            + 0.30 * (df["global_dysregulation_score"] - 0.5)
            + 0.20 * self.rng.normal(0.0, 1.0, len(df))
        )
        flare_prob = 1.0 / (1.0 + np.exp(-(flare_logit - self.cfg.flare_threshold_logit)))
        crash_prob = 1.0 / (1.0 + np.exp(-(flare_logit - self.cfg.crash_threshold_logit)))

        last_day_mask = df["next_true_regime_id"].isna()
        flare_draws = self.rng.random(len(df))
        crash_draws = self.rng.random(len(df))
        df["flare_next_day"] = pd.array(np.where(flare_draws < flare_prob, 1, 0), dtype="Int64")
        df["crash_next_day"] = pd.array(np.where(crash_draws < crash_prob, 1, 0), dtype="Int64")
        df.loc[last_day_mask, ["flare_next_day", "crash_next_day"]] = pd.NA

        df["next_day_fatigue"] = df.groupby("subject_id")["fatigue"].shift(-1)
        df["next_day_pain"] = df.groupby("subject_id")["pain"].shift(-1)
        score_p2 = df.groupby("subject_id")["global_dysregulation_score"].shift(-2)
        df["worsening_2day"] = ((score_p2 - df["global_dysregulation_score"]) >= 0.18).astype("Int64")
        df.loc[df["next_day_score"].isna(), "next_day_score"] = np.nan
        df = df.drop(columns=[c for c in df.columns if c.startswith("next_") and c not in {
            "next_day_score", "next_day_fatigue", "next_day_pain",
            "next_true_regime_id", "next_dwell_time",
            "next_latent_instability", "next_delayed_stress_load", "next_delayed_recovery_load",
        }])
        return df

    @staticmethod
    def _global_score(df: pd.DataFrame) -> pd.Series:
        """Today's linear-in-observations severity score. Bounded to [0, 1]."""
        raw = (
            0.20 * df["fatigue"]
            + 0.17 * df["pain"]
            + 0.18 * (10.0 - df["sleep_quality"])
            + 0.10 * (10.0 - df["mood_calm"])
            + 0.10 * (10.0 - df["energy"])
            + 0.10 * df["heaviness"]
            + 0.08 * (10.0 - df["appetite"])
            + 0.07 * (5.0 - df["bowel_quality"]) * 2.0
        ) / 10.0
        return raw.clip(0.0, 1.0)

    # ---- labels -------------------------------------------------------------

    def _add_semantic_labels(self, visits: pd.DataFrame, latent: pd.DataFrame) -> pd.DataFrame:
        """Generate noisy practitioner-style labels from observations + bias."""
        df = visits.merge(latent, on=["subject_id", "day"], how="left")
        qi_labels: List[str] = []
        tcm_labels: List[str] = []
        signatures: List[str] = []

        for row in df.itertuples(index=False):
            obs = {name: getattr(row, name) for name in OBSERVATION_NAMES}
            cold_hot = obs["cold_hot"]
            severity = float(getattr(row, "global_dysregulation_score"))
            qi, tcm = self._practitioner_rule(obs, severity)
            sig = self._signature_for(tcm, obs, severity)
            qi_labels.append(qi)
            tcm_labels.append(tcm)
            signatures.append(sig)

        df["qi_like_label"] = qi_labels
        df["tcm_like_label"] = tcm_labels
        df["contrarian_signature"] = signatures
        df = self._inject_label_noise(df)
        return df

    def _practitioner_rule(self, obs: Dict[str, float], severity: float) -> Tuple[str, str]:
        """Map noisy observations to a (qi, tcm) label using thresholded rules."""
        fatigue = obs["fatigue"]
        pain = obs["pain"]
        sleep = obs["sleep_quality"]
        mood = obs["mood_calm"]
        appetite = obs["appetite"]
        bowel = obs["bowel_quality"]
        energy = obs["energy"]
        heaviness = obs["heaviness"]
        cold_hot = obs["cold_hot"]

        bias = self.cfg.practitioner_bias
        bias_noise = float(self.rng.normal(0.0, 0.5))

        if severity < 0.18 + 0.05 * bias_noise:
            qi = "balanced_flow"
        elif fatigue > 5.5 - bias and appetite < 6.0:
            qi = "depleted"
        elif mood < 5.5 - bias and pain < 5.0:
            qi = "stuck_agitated"
        elif heaviness > 5.0 - bias and appetite < 6.0:
            qi = "heavy_damp_like"
        elif pain > 5.0 - bias and (energy < 6.0 or cold_hot > 6.0):
            qi = "hot_overactive"
        else:
            qi = "balanced_flow"

        if fatigue > 5.5 - bias and bowel < 3.4 and appetite < 6.5:
            tcm = "spleen_qi_deficiency_like"
        elif mood < 5.5 - bias and sleep < 6.5:
            tcm = "liver_qi_stagnation_like"
        elif (pain > 4.8 - bias or cold_hot > 6.0) and heaviness > 4.5 - bias:
            tcm = "damp_heat_like"
        elif fatigue > 4.5 and cold_hot > 6.5 - bias and energy < 6.0:
            tcm = "yin_deficiency_like"
        else:
            tcm = "mixed_pattern_like"
        return qi, tcm

    def _signature_for(self, tcm_label: str, obs: Dict[str, float], severity: float) -> str:
        """Tag rows with intentional split/merge signatures."""
        if tcm_label == "spleen_qi_deficiency_like":
            return "spleen_like_subtype_ruminative" if obs["mood_calm"] < 5.5 else "spleen_like_subtype_digestive"
        if tcm_label in {"liver_qi_stagnation_like", "damp_heat_like"}:
            if 4.0 < obs["pain"] < 7.0 and 4.0 < obs["heaviness"] < 7.0:
                return "merged_stress_inflammation_bridge"
        return "canonical_single_cluster"

    def _inject_label_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Randomly perturb label assignments per the difficulty profile."""
        rate = float(np.clip(self.cfg.label_noise_rate, 0.0, 1.0))
        if rate <= 0.0 or df.empty:
            return df
        out = df.copy()
        qi_choices = np.array(sorted(out["qi_like_label"].dropna().unique()))
        tcm_choices = np.array(sorted(out["tcm_like_label"].dropna().unique()))
        mask = self.rng.random(len(out)) < rate
        if len(qi_choices):
            out.loc[mask, "qi_like_label"] = self.rng.choice(qi_choices, size=int(mask.sum()))
        if len(tcm_choices):
            out.loc[mask, "tcm_like_label"] = self.rng.choice(tcm_choices, size=int(mask.sum()))
        out.loc[mask, "contrarian_signature"] = "label_noise_injected"
        return out

    # ---- missingness, attractor summary, outputs ----------------------------

    def _inject_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Random missingness with elevated rate inside stuck regimes."""
        out = df.copy()
        in_stuck = out["attractor_state"].fillna(0).astype(int).to_numpy(bool)
        rate = float(self.cfg.missing_rate)
        severe_rate = float(np.clip(rate * self.cfg.missing_rate_severe_multiplier, 0.0, 0.6))
        for col in OBSERVATION_NAMES:
            base = self.rng.random(len(out)) < rate
            severe = self.rng.random(len(out)) < severe_rate
            mask = np.where(in_stuck, severe, base)
            out.loc[mask, col] = np.nan
        return out

    def _summarize_attractors(self, visits: pd.DataFrame) -> pd.DataFrame:
        """Per-subject stuck-state occupancy and dwell summary."""
        rows: List[Dict] = []
        for sid, sub in visits.groupby("subject_id"):
            regimes = sub["true_regime_id"].to_numpy()
            dwell = sub["dwell_time"].to_numpy()
            n = len(regimes)
            frac_dep = float(np.mean(regimes == 4)) if n else 0.0
            frac_ag = float(np.mean(regimes == 5)) if n else 0.0
            longest = int(np.max(dwell[regimes == 4]) if np.any(regimes == 4) else 0)
            longest_ag = int(np.max(dwell[regimes == 5]) if np.any(regimes == 5) else 0)
            n_entries = int(np.sum((regimes[1:] != regimes[:-1]) & np.isin(regimes[1:], STUCK_REGIME_IDS))) if n > 1 else 0
            rows.append({
                "subject_id": int(sid),
                "frac_in_stuck_depleted": frac_dep,
                "frac_in_stuck_agitated": frac_ag,
                "frac_in_any_stuck": frac_dep + frac_ag,
                "longest_stuck_depleted_dwell": longest,
                "longest_stuck_agitated_dwell": longest_ag,
                "n_stuck_entries": n_entries,
            })
        return pd.DataFrame(rows)

    def _write_outputs(
        self,
        subjects: pd.DataFrame,
        visits: pd.DataFrame,
        latent: pd.DataFrame,
        events: pd.DataFrame,
        attractors: pd.DataFrame,
    ) -> None:
        """Write all CSVs, JSON metadata, and ground-truth registries."""
        subjects.to_csv(self.output_dir / "subjects.csv", index=False)
        visits.to_csv(self.output_dir / "visits.csv", index=False)
        latent.to_csv(self.output_dir / "latent_states.csv", index=False)
        events.to_csv(self.output_dir / "events.csv", index=False)

        # true_regimes.csv: regime registry
        regime_rows = []
        for rid, name in enumerate(REGIME_NAMES):
            regime_rows.append({
                "regime_id": rid,
                "regime_name": name,
                "is_stuck": rid in STUCK_REGIME_IDS,
                "is_active": rid in ACTIVE_REGIME_IDS,
                "prototype_v": float(REGIME_PROTOTYPES[rid, 0]),
                "prototype_s": float(REGIME_PROTOTYPES[rid, 1]),
                "prototype_i": float(REGIME_PROTOTYPES[rid, 2]),
                "prototype_d": float(REGIME_PROTOTYPES[rid, 3]),
                "drift_strength": float(REGIME_DRIFT_STRENGTH[rid]),
                "alias_group": ALIAS_GROUPS[rid],
                "base_self_loop": float(self.transition_matrix[rid, rid]),
            })
        pd.DataFrame(regime_rows).to_csv(self.output_dir / "true_regimes.csv", index=False)

        # true_transition_matrices.csv: base + per-subject bias as long format
        trans_rows: List[Dict] = []
        for i in range(N_REGIMES):
            for j in range(N_REGIMES):
                trans_rows.append({
                    "subject_id": -1,
                    "from_regime_id": i,
                    "to_regime_id": j,
                    "base_prob": float(self.transition_matrix[i, j]),
                    "bias_logit": 0.0,
                })
        for sid, bias in self.subject_bias_matrices.items():
            for i in range(N_REGIMES):
                for j in range(N_REGIMES):
                    trans_rows.append({
                        "subject_id": int(sid),
                        "from_regime_id": i,
                        "to_regime_id": j,
                        "base_prob": float(self.transition_matrix[i, j]),
                        "bias_logit": float(bias[i, j]),
                    })
        pd.DataFrame(trans_rows).to_csv(self.output_dir / "true_transition_matrices.csv", index=False)

        attractors.to_csv(self.output_dir / "true_attractor_states.csv", index=False)

        metadata = {
            "config": asdict(self.cfg),
            "latent_names": LATENT_NAMES,
            "observation_names": OBSERVATION_NAMES,
            "event_types": EVENT_TYPES,
            "regime_names": REGIME_NAMES,
            "stuck_regime_ids": STUCK_REGIME_IDS,
            "alias_groups": ALIAS_GROUPS,
            "ground_truth_files": {
                "subjects.csv": "Subject-level metadata: hidden_subtype, sensitivities, attractor_susceptibility, recovery_efficiency.",
                "visits.csv": "Per-(subject, day) observations, true_regime_id, dwell_time, latent_instability, delayed_*_load, outcomes.",
                "latent_states.csv": "Continuous latent z_t per (subject, day).",
                "events.csv": "Stress/recovery/treatment event log per (subject, day).",
                "true_regimes.csv": "Regime registry: id, name, prototype z, drift strength, alias group, self-loop probability.",
                "true_transition_matrices.csv": "Base 7x7 transition matrix and per-subject additive logit bias matrices.",
                "true_attractor_states.csv": "Per-subject occupancy fraction and longest dwell in stuck regimes.",
            },
            "design_notes": {
                "model": "Switching state-space: discrete regime c_t in 7 categories + continuous latent z_t in R^4.",
                "subtype_role": "hidden_subtype shifts a subject's transition-bias matrix (dynamic property), not just baseline z.",
                "stuck_attractors": "stuck_depleted (4) and stuck_agitated (5) have high self-transition probability scaled by sticky_strength.",
                "delayed_effects": "Stress/treatment/recovery loads decay exponentially (load_decay) and modulate regime transitions and flare risk.",
                "observation_aliasing": "Regimes are grouped (calm/wired/hot/heavy) and share an observation offset, so different regimes can produce similar x_t.",
                "outcomes": (
                    "next_day_score and flare_next_day depend on c_{t+1}, dwell_time_{t+1}, latent_instability, and delayed loads, "
                    "with a small linear-in-observations component. Today's global_dysregulation_score alone is a weak predictor."
                ),
                "labels": "TCM-like labels are thresholded functions of NOISY observations with practitioner bias and label noise (not direct from regimes).",
                "ontology_mismatch": "Aliased regimes (e.g. stuck_agitated vs stressed_recoverable) can yield the same label; same regime can yield different labels via noise.",
            },
            "framing": (
                "Synthetic benchmark for latent-state recovery and ontology mismatch detection. "
                "Not a biological simulator. Not evidence for TCM or Qi."
            ),
            "recommended_tasks": [
                "Recover discrete regime c_t from observations only.",
                "Detect stuck-attractor states via self-resonance.",
                "Predict next_day_score / flare_next_day with subject-level dynamics, not just today's severity.",
                "Compare TCM-like labels against true_regime to surface ontology mismatch.",
                "Recover hidden_subtype from subject-level transition fingerprints.",
            ],
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def summarize_dataset(visits: pd.DataFrame) -> pd.DataFrame:
    """Return a concise per-column summary for printing."""
    numeric = visits.select_dtypes(include=[np.number])
    return pd.DataFrame({
        "non_null": visits.notna().sum(),
        "mean": numeric.mean(),
        "std": numeric.std(),
    })


def parse_args() -> GeneratorConfig:
    """Parse CLI into a GeneratorConfig."""
    parser = argparse.ArgumentParser(description="Generate a regime-switching synthetic GRM-TCM benchmark dataset.")
    parser.add_argument("--output-dir", default="synthetic_grm_tcm")
    parser.add_argument("--difficulty", default="medium", choices=sorted(DIFFICULTY_PRESETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-subjects", type=int, default=80)
    parser.add_argument("--n-days", type=int, default=60)
    args = parser.parse_args()
    return GeneratorConfig(
        n_subjects=args.n_subjects,
        n_days=args.n_days,
        random_seed=args.seed,
        output_dir=args.output_dir,
        difficulty=args.difficulty,
    )


if __name__ == "__main__":
    cfg = parse_args()
    gen = SyntheticGRMTCMGenerator(cfg)
    outputs = gen.run()
    print(f"Wrote synthetic dataset to: {gen.output_dir.resolve()}")
    print("Generated files: subjects.csv, visits.csv, latent_states.csv, events.csv, "
          "true_regimes.csv, true_transition_matrices.csv, true_attractor_states.csv, metadata.json")
    visits = outputs["visits"]
    regime_counts = visits["true_regime"].value_counts(normalize=True).round(3)
    print("\nRegime occupancy:")
    print(regime_counts.to_string())
    flare_rate = float(visits["flare_next_day"].dropna().astype(int).mean())
    crash_rate = float(visits["crash_next_day"].dropna().astype(int).mean())
    print(f"\nFlare rate: {flare_rate:.3f}    Crash rate: {crash_rate:.3f}")
    print(summarize_dataset(visits).head(20))
