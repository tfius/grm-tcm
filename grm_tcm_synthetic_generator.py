from __future__ import annotations

"""
GRM-TCM synthetic dataset generator.

Creates a controlled longitudinal benchmark with:
- hidden whole-body latent states
- symptom / physiology observations
- stress, recovery, and treatment events
- predictable next-day outcomes
- naive Qi-like and TCM-like labels
- intentional ontology mismatch / contrarian signatures

Run:
  python grm_tcm_synthetic_generator.py

Outputs:
  synthetic_grm_tcm/subjects.csv
  synthetic_grm_tcm/visits.csv
  synthetic_grm_tcm/latent_states.csv
  synthetic_grm_tcm/events.csv
  synthetic_grm_tcm/metadata.json
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LATENT_NAMES = [
    "vitality_depletion",
    "stress_activation",
    "inflammatory_load",
    "digestive_instability",
]

OBSERVATION_NAMES = [
    "sleep_quality",
    "hrv",
    "resting_hr",
    "body_temp",
    "fatigue",
    "pain",
    "appetite",
    "bowel_quality",
    "mood_calm",
    "energy",
    "heaviness",
    "cold_hot",
]

EVENT_TYPES = ["stress_event", "recovery_event", "treatment_event"]


@dataclass
class GeneratorConfig:
    n_subjects: int = 80
    n_days: int = 60
    latent_dim: int = 4
    random_seed: int = 42
    output_dir: str = "synthetic_grm_tcm"

    latent_noise_std: float = 0.18
    obs_noise_std: float = 0.45
    missing_rate: float = 0.03

    stress_event_rate: float = 0.10
    recovery_event_rate: float = 0.07
    treatment_event_rate: float = 0.12

    flare_threshold: float = 0.65
    crash_threshold: float = 0.85

    hidden_subtype_strength: float = 0.45
    split_spleen_like_label: bool = True
    merge_liver_damp_like_labels: bool = True


class SyntheticGRMTCMGenerator:
    def __init__(self, config: GeneratorConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.random_seed)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stable but cross-coupled latent dynamics.
        self.A = np.array([
            [0.87, 0.10, 0.08, 0.06],
            [0.07, 0.83, 0.12, 0.04],
            [0.05, 0.08, 0.86, 0.09],
            [0.06, 0.05, 0.10, 0.84],
        ])

        self.B_events = {
            "stress_event": np.array([0.18, 0.45, 0.16, 0.10]),
            "recovery_event": np.array([-0.20, -0.18, -0.12, -0.10]),
            "treatment_event": np.array([-0.15, -0.10, -0.20, -0.16]),
        }

        # Hidden treatment-response subtypes.
        self.treatment_prototypes = {
            0: np.array([-0.28, -0.08, -0.10, -0.18]),  # digestive responder
            1: np.array([-0.14, -0.24, -0.16, -0.06]),  # stress/pain responder
            2: np.array([-0.10, -0.06, -0.28, -0.12]),  # inflammatory responder
        }

        # Observation matrix: rows are observed variables, cols are latent variables.
        self.C = np.array([
            [-0.35, -0.18, -0.12, -0.10],  # sleep_quality
            [-0.22, -0.38, -0.18, -0.08],  # hrv
            [0.10, 0.28, 0.14, 0.04],      # resting_hr
            [0.02, 0.06, 0.25, 0.02],      # body_temp
            [0.42, 0.18, 0.20, 0.12],      # fatigue
            [0.15, 0.18, 0.40, 0.08],      # pain
            [-0.10, -0.08, -0.10, -0.34],  # appetite
            [-0.08, -0.04, -0.08, -0.40],  # bowel_quality
            [-0.16, -0.42, -0.10, -0.04],  # mood_calm
            [-0.46, -0.16, -0.14, -0.10],  # energy
            [0.18, 0.05, 0.16, 0.36],      # heaviness
            [-0.10, 0.08, 0.35, 0.02],     # cold_hot
        ])

        self.obs_baseline = {
            "sleep_quality": 7.0, "hrv": 58.0, "resting_hr": 66.0,
            "body_temp": 36.7, "fatigue": 3.5, "pain": 2.8,
            "appetite": 7.0, "bowel_quality": 3.9, "mood_calm": 6.8,
            "energy": 6.8, "heaviness": 3.4, "cold_hot": 5.0,
        }
        self.obs_scale = {
            "sleep_quality": 1.5, "hrv": 11.0, "resting_hr": 8.0,
            "body_temp": 0.35, "fatigue": 2.2, "pain": 2.0,
            "appetite": 1.8, "bowel_quality": 1.0, "mood_calm": 1.8,
            "energy": 2.0, "heaviness": 2.0, "cold_hot": 1.5,
        }

    def run(self) -> Dict[str, pd.DataFrame]:
        subjects = self._generate_subjects()
        visits, latent, events = self._simulate(subjects)
        visits = self._add_outcomes(visits)
        visits = self._add_semantic_labels(visits, latent)
        visits = self._inject_missingness(visits)
        self._write_outputs(subjects, visits, latent, events)
        return {"subjects": subjects, "visits": visits, "latent_states": latent, "events": events}

    def _generate_subjects(self) -> pd.DataFrame:
        rows = []
        for sid in range(self.cfg.n_subjects):
            subtype = int(self.rng.integers(0, 3))
            baseline = self.rng.normal(0.0, 0.45, self.cfg.latent_dim)
            subtype_offset = np.zeros(self.cfg.latent_dim)
            if subtype == 0:
                subtype_offset = np.array([0.25, -0.05, 0.05, 0.35])
            elif subtype == 1:
                subtype_offset = np.array([0.28, 0.22, 0.08, 0.06])
            else:
                subtype_offset = np.array([0.08, 0.26, 0.30, 0.02])
            subtype_offset *= self.cfg.hidden_subtype_strength
            z0 = baseline + subtype_offset
            rows.append({
                "subject_id": sid,
                "hidden_subtype": subtype,
                "baseline_vitality_depletion": z0[0],
                "baseline_stress_activation": z0[1],
                "baseline_inflammatory_load": z0[2],
                "baseline_digestive_instability": z0[3],
                "sensitivity_stress": self.rng.uniform(0.7, 1.3),
                "sensitivity_recovery": self.rng.uniform(0.7, 1.2),
                "sensitivity_treatment": self.rng.uniform(0.7, 1.4),
                "chronic_load": self.rng.uniform(0.0, 0.8),
            })
        return pd.DataFrame(rows)

    def _simulate(self, subjects: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        visit_rows: List[Dict] = []
        latent_rows: List[Dict] = []
        event_rows: List[Dict] = []

        for subj in subjects.itertuples(index=False):
            sid = int(subj.subject_id)
            subtype = int(subj.hidden_subtype)
            z = np.array([
                subj.baseline_vitality_depletion,
                subj.baseline_stress_activation,
                subj.baseline_inflammatory_load,
                subj.baseline_digestive_instability,
            ], dtype=float)
            treatment_proto = self.treatment_prototypes[subtype]

            for day in range(self.cfg.n_days):
                u = np.zeros(self.cfg.latent_dim)
                for event in self._sample_events():
                    effect = self.B_events[event].copy()
                    if event == "stress_event":
                        effect *= subj.sensitivity_stress
                    elif event == "recovery_event":
                        effect *= subj.sensitivity_recovery
                    elif event == "treatment_event":
                        effect = effect * subj.sensitivity_treatment + treatment_proto
                    u += effect
                    event_rows.append({"subject_id": sid, "day": day, "event_type": event})

                weekly_rhythm = 0.06 * np.sin(2 * np.pi * day / 7.0)
                chronic_push = np.array([
                    0.030 * subj.chronic_load,
                    0.020 * subj.chronic_load,
                    0.025 * subj.chronic_load,
                    0.020 * subj.chronic_load,
                ])

                z = self.A @ z + u + chronic_push + np.array([0.02, weekly_rhythm, 0.01, 0.015])
                z += self.rng.normal(0.0, self.cfg.latent_noise_std, self.cfg.latent_dim)

                # Mild nonlinear amplification.
                if z[1] > 1.0 and z[2] > 0.9:
                    z[2] += 0.08
                if z[0] > 1.0 and z[3] > 0.9:
                    z[0] += 0.05
                    z[3] += 0.04

                latent_rows.append({
                    "subject_id": sid,
                    "day": day,
                    **{name: float(z[i]) for i, name in enumerate(LATENT_NAMES)},
                })
                visit_rows.append({"subject_id": sid, "day": day, **self._latent_to_observations(z)})

        return pd.DataFrame(visit_rows), pd.DataFrame(latent_rows), pd.DataFrame(event_rows)

    def _sample_events(self) -> List[str]:
        events = []
        if self.rng.random() < self.cfg.stress_event_rate:
            events.append("stress_event")
        if self.rng.random() < self.cfg.recovery_event_rate:
            events.append("recovery_event")
        if self.rng.random() < self.cfg.treatment_event_rate:
            events.append("treatment_event")
        return events

    def _latent_to_observations(self, z: np.ndarray) -> Dict[str, float]:
        raw = self.C @ z + self.rng.normal(0.0, self.cfg.obs_noise_std, len(OBSERVATION_NAMES))
        out = {}
        for i, name in enumerate(OBSERVATION_NAMES):
            val = self.obs_baseline[name] + self.obs_scale[name] * raw[i]
            out[name] = self._clip(name, val)
        return out

    @staticmethod
    def _clip(name: str, value: float) -> float:
        ranges = {
            "sleep_quality": (1.0, 10.0), "hrv": (10.0, 120.0),
            "resting_hr": (40.0, 120.0), "body_temp": (35.2, 39.5),
            "fatigue": (0.0, 10.0), "pain": (0.0, 10.0),
            "appetite": (0.0, 10.0), "bowel_quality": (1.0, 5.0),
            "mood_calm": (0.0, 10.0), "energy": (0.0, 10.0),
            "heaviness": (0.0, 10.0), "cold_hot": (0.0, 10.0),
        }
        lo, hi = ranges[name]
        return float(np.clip(value, lo, hi))

    def _add_outcomes(self, visits: pd.DataFrame) -> pd.DataFrame:
        df = visits.sort_values(["subject_id", "day"]).reset_index(drop=True).copy()
        df["global_dysregulation_score"] = (
            0.20 * df["fatigue"]
            + 0.17 * df["pain"]
            + 0.18 * (10.0 - df["sleep_quality"])
            + 0.10 * (10.0 - df["mood_calm"])
            + 0.10 * (10.0 - df["energy"])
            + 0.10 * df["heaviness"]
            + 0.08 * (10.0 - df["appetite"])
            + 0.07 * (5.0 - df["bowel_quality"]) * 2.0
        ) / 10.0
        df["next_day_score"] = df.groupby("subject_id")["global_dysregulation_score"].shift(-1)
        df["next_day_fatigue"] = df.groupby("subject_id")["fatigue"].shift(-1)
        df["next_day_pain"] = df.groupby("subject_id")["pain"].shift(-1)
        df["flare_next_day"] = (df["next_day_score"] >= self.cfg.flare_threshold).astype("Int64")
        df["crash_next_day"] = (df["next_day_score"] >= self.cfg.crash_threshold).astype("Int64")
        score_p2 = df.groupby("subject_id")["global_dysregulation_score"].shift(-2)
        df["worsening_2day"] = ((score_p2 - df["global_dysregulation_score"]) >= 0.18).astype("Int64")
        return df

    def _add_semantic_labels(self, visits: pd.DataFrame, latent: pd.DataFrame) -> pd.DataFrame:
        df = visits.merge(latent, on=["subject_id", "day"], how="left")
        qi_labels, tcm_labels, signatures = [], [], []

        for row in df.itertuples(index=False):
            vd = getattr(row, "vitality_depletion")
            sa = getattr(row, "stress_activation")
            il = getattr(row, "inflammatory_load")
            di = getattr(row, "digestive_instability")
            cold_hot = getattr(row, "cold_hot")

            if vd < 0.15 and sa < 0.15 and il < 0.15 and di < 0.15:
                qi = "balanced_flow"
            elif vd > 0.70 and di > 0.45:
                qi = "depleted"
            elif sa > 0.70 and il < 0.55:
                qi = "stuck_agitated"
            elif di > 0.75:
                qi = "heavy_damp_like"
            elif il > 0.75 and sa > 0.45:
                qi = "hot_overactive"
            else:
                qi = "balanced_flow"

            if vd > 0.65 and di > 0.45:
                tcm = "spleen_qi_deficiency_like"
            elif sa > 0.65 and il < 0.55:
                tcm = "liver_qi_stagnation_like"
            elif il > 0.60 and di > 0.40:
                tcm = "damp_heat_like"
            elif vd > 0.60 and il > 0.55 and cold_hot > 6.0:
                tcm = "yin_deficiency_like"
            else:
                tcm = "mixed_pattern_like"

            if self.cfg.split_spleen_like_label and tcm == "spleen_qi_deficiency_like":
                sig = "spleen_like_subtype_ruminative" if sa > 0.30 else "spleen_like_subtype_digestive"
            elif self.cfg.merge_liver_damp_like_labels and tcm in {"liver_qi_stagnation_like", "damp_heat_like"}:
                sig = "merged_stress_inflammation_bridge" if 0.45 < sa < 0.80 and 0.35 < il < 0.80 else "canonical_single_cluster"
            else:
                sig = "canonical_single_cluster"

            qi_labels.append(qi)
            tcm_labels.append(tcm)
            signatures.append(sig)

        df["qi_like_label"] = qi_labels
        df["tcm_like_label"] = tcm_labels
        df["contrarian_signature"] = signatures
        return df

    def _inject_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = OBSERVATION_NAMES
        mask = self.rng.random((len(out), len(cols))) < self.cfg.missing_rate
        for j, col in enumerate(cols):
            out.loc[mask[:, j], col] = np.nan
        return out

    def _write_outputs(self, subjects: pd.DataFrame, visits: pd.DataFrame, latent: pd.DataFrame, events: pd.DataFrame) -> None:
        subjects.to_csv(self.output_dir / "subjects.csv", index=False)
        visits.to_csv(self.output_dir / "visits.csv", index=False)
        latent.to_csv(self.output_dir / "latent_states.csv", index=False)
        events.to_csv(self.output_dir / "events.csv", index=False)
        metadata = {
            "config": asdict(self.cfg),
            "latent_names": LATENT_NAMES,
            "observation_names": OBSERVATION_NAMES,
            "event_types": EVENT_TYPES,
            "notes": {
                "purpose": "Synthetic benchmark for GRM-style latent-state modeling of integrative physiology.",
                "not_biology": "This is not a biological simulator or proof of TCM. It is a controlled benchmark.",
                "contrarian_examples": [
                    "One syndrome-like label is split into hidden subtypes.",
                    "Some labels merge into a bridge state.",
                    "Treatment response depends partly on hidden subtype, not label."
                ],
                "recommended_tasks": [
                    "Predict next_day_score.",
                    "Predict flare_next_day.",
                    "Recover latent state from observations only.",
                    "Detect mismatch between latent clusters and semantic labels."
                ]
            }
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def summarize_dataset(visits: pd.DataFrame) -> pd.DataFrame:
    numeric = visits.select_dtypes(include=[np.number])
    return pd.DataFrame({"non_null": visits.notna().sum(), "mean": numeric.mean(), "std": numeric.std()})


if __name__ == "__main__":
    cfg = GeneratorConfig()
    gen = SyntheticGRMTCMGenerator(cfg)
    outputs = gen.run()
    print(f"Wrote synthetic dataset to: {gen.output_dir.resolve()}")
    print("Generated files: subjects.csv, visits.csv, latent_states.csv, events.csv, metadata.json")
    print(summarize_dataset(outputs["visits"]).head(20))
