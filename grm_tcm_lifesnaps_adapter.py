"""LifeSnaps dataset adapter for GRM-TCM pipeline.

Converts LifeSnaps (71 subjects × 4 months, Fitbit Sense + EMA + surveys)
into visits.csv / subjects.csv / events.csv for grm_tcm_train.py.

Source: https://zenodo.org/records/7229547 (CC-BY 4.0, no credentials)

Usage:
  unzip rais_anonymized.zip "*.csv" -d /tmp/lifesnaps
  python grm_tcm_lifesnaps_adapter.py --input-dir /tmp/lifesnaps/rais_anonymized
  python grm_tcm_train.py --input-dir lifesnaps_grm_tcm --graph-feature-source takens --n-modes 8 --rho 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Fitbit physiology columns to use as observations.
PHYSIO_COLS = [
    "resting_hr", "bpm", "rmssd", "stress_score",
    "sleep_duration", "sleep_efficiency", "sleep_deep_ratio",
    "sleep_rem_ratio", "sleep_light_ratio", "sleep_wake_ratio",
    "steps", "calories", "nightly_temperature",
    "full_sleep_breathing_rate",
    "lightly_active_minutes", "moderately_active_minutes",
    "very_active_minutes", "sedentary_minutes",
]

# EMA mood columns (binary/continuous self-report).
MOOD_COLS = [
    "ALERT", "HAPPY", "NEUTRAL", "RESTED/RELAXED",
    "SAD", "TENSE/ANXIOUS", "TIRED",
]


def convert_lifesnaps(input_dir: str, output_dir: str, min_days: int = 30) -> None:
    """Convert LifeSnaps to GRM-TCM pipeline format."""
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load daily data.
    daily_path = inp / "csv_rais_anonymized" / "daily_fitbit_sema_df_unprocessed.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing {daily_path}")
    df = pd.read_csv(daily_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "id"])
    print(f"Loaded {len(df)} daily rows, {df['id'].nunique()} users")

    # Map user IDs to integers.
    uid_map = {uid: i for i, uid in enumerate(sorted(df["id"].unique()))}
    df["subject_id"] = df["id"].map(uid_map)

    # Filter to users with enough days.
    day_counts = df.groupby("subject_id")["date"].nunique()
    valid = day_counts[day_counts >= min_days].index
    df = df[df["subject_id"].isin(valid)]
    print(f"  {len(valid)} users with >= {min_days} days")

    # Day offset per subject.
    df = df.sort_values(["subject_id", "date"]).reset_index(drop=True)
    df["day"] = df.groupby("subject_id")["date"].transform(
        lambda x: (x - x.min()).dt.days
    )
    df["visit_id"] = np.arange(len(df))

    # Select observation columns.
    obs_cols = [c for c in PHYSIO_COLS if c in df.columns]

    # Rename mood columns (strip special chars).
    mood_rename = {}
    for c in MOOD_COLS:
        if c in df.columns:
            clean = "mood_" + c.lower().replace("/", "_").replace(" ", "_")
            mood_rename[c] = clean
            obs_cols.append(clean)
    df = df.rename(columns=mood_rename)

    # Convert numeric.
    for c in obs_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dysregulation score: composite of negative mood + stress + poor sleep.
    neg_cols = [c for c in ["mood_sad", "mood_tense_anxious", "mood_tired", "stress_score"]
                if c in df.columns]
    pos_cols = [c for c in ["mood_happy", "mood_alert", "mood_rested_relaxed", "sleep_efficiency"]
                if c in df.columns]
    parts = []
    for c in neg_cols:
        cmin, cmax = df[c].min(), df[c].max()
        if pd.notna(cmax) and cmax > cmin:
            parts.append((df[c] - cmin) / (cmax - cmin))
    for c in pos_cols:
        cmin, cmax = df[c].min(), df[c].max()
        if pd.notna(cmax) and cmax > cmin:
            parts.append(1.0 - (df[c] - cmin) / (cmax - cmin))
    if parts:
        df["global_dysregulation_score"] = np.nanmean(
            np.column_stack([s.to_numpy() for s in parts]), axis=1
        )
    else:
        df["global_dysregulation_score"] = np.nan

    # Next-day targets.
    df["next_day_score"] = df.groupby("subject_id")["global_dysregulation_score"].shift(-1)
    med = df["global_dysregulation_score"].median()
    df["flare_next_day"] = (df["next_day_score"] > med).astype(float)
    df.loc[df["next_day_score"].isna(), "flare_next_day"] = np.nan

    # Events: high-exertion days as stressor events.
    events = []
    if "very_active_minutes" in df.columns:
        threshold = df["very_active_minutes"].quantile(0.75)
        for _, row in df[df["very_active_minutes"] > threshold].iterrows():
            events.append({
                "subject_id": int(row["subject_id"]),
                "day": int(row["day"]),
                "event_type": "treatment_event",
            })

    # Load personality surveys as constitution proxy.
    personality_path = inp / "scored_surveys" / "personality.csv"
    subjects_extra = {}
    if personality_path.exists():
        pers = pd.read_csv(personality_path)
        pers_cols = ["extraversion", "agreeableness", "conscientiousness",
                     "stability", "intellect"]
        if "user_id" in pers.columns and all(c in pers.columns for c in pers_cols):
            pers["subject_id"] = pers["user_id"].map(uid_map)
            pers = pers.dropna(subset=["subject_id"])
            # Take first survey per user.
            pers = pers.sort_values("submitdate").groupby("subject_id").first().reset_index()
            for _, row in pers.iterrows():
                sid = int(row["subject_id"])
                subjects_extra[sid] = {
                    f"constitution_{c}": float(row[c])
                    for c in pers_cols if pd.notna(row[c])
                }

    # Build subjects.csv.
    subjects = []
    for sid in sorted(df["subject_id"].unique()):
        entry = {"subject_id": sid}
        sub = df[df["subject_id"] == sid].iloc[0]
        for c in ["age", "gender", "bmi"]:
            if c in df.columns and pd.notna(sub[c]):
                entry[c] = sub[c]
        if sid in subjects_extra:
            entry.update(subjects_extra[sid])
        subjects.append(entry)

    # Write.
    keep = ["visit_id", "subject_id", "day"] + obs_cols + [
        "global_dysregulation_score", "next_day_score", "flare_next_day",
    ]
    df[[c for c in keep if c in df.columns]].to_csv(out / "visits.csv", index=False)
    pd.DataFrame(subjects).to_csv(out / "subjects.csv", index=False)
    if events:
        pd.DataFrame(events).to_csv(out / "events.csv", index=False)

    print(f"\nLifeSnaps adapter complete.")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Visits:   {len(df)}")
    print(f"  Features: {len(obs_cols)} ({obs_cols[:6]}...)")
    print(f"  Dysreg coverage: {df['global_dysregulation_score'].notna().mean():.1%}")
    print(f"  Events: {len(events)} high-exertion days")
    print(f"  Constitution: {len(subjects_extra)} users with Big Five scores")
    print(f"  Output: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LifeSnaps to GRM-TCM format.")
    parser.add_argument("--input-dir", default="/tmp/lifesnaps/rais_anonymized")
    parser.add_argument("--output-dir", default="lifesnaps_grm_tcm")
    parser.add_argument("--min-days", type=int, default=30)
    args = parser.parse_args()
    convert_lifesnaps(args.input_dir, args.output_dir, args.min_days)
