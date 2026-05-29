"""Flaredown autoimmune symptom tracker adapter for GRM-TCM pipeline.

Converts the Flaredown Kaggle dataset (long-format, ~8M rows, 17K users)
into visits.csv / subjects.csv / events.csv for grm_tcm_train.py.

Source: kaggle.com/datasets/flaredown/flaredown-autoimmune-symptom-tracker

Usage:
  python grm_tcm_flaredown_adapter.py --input flares-export.csv --output-dir flaredown_grm_tcm
  python grm_tcm_train.py --input-dir flaredown_grm_tcm --graph-feature-source takens --n-modes 8 --rho 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Top symptoms to pivot into observation columns (by frequency).
TOP_SYMPTOMS = [
    "Headache", "Fatigue", "Nausea", "Joint pain", "Stomach Pain",
    "Diarrhea", "Anxiety", "Dizziness", "Brain fog", "Depression",
    "Insomnia", "Back pain", "Bloating",
]

# Weather features to include.
WEATHER_FEATURES = [
    "temperature_min", "temperature_max", "humidity", "pressure",
]


def convert_flaredown(
    input_path: str,
    output_dir: str,
    min_days: int = 60,
    max_subjects: int = 500,
) -> None:
    """Convert Flaredown long-format CSV to GRM-TCM pipeline format."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading Flaredown data (may take a moment)...")
    df = pd.read_csv(input_path, low_memory=False)
    df["checkin_date"] = pd.to_datetime(df["checkin_date"], errors="coerce")
    df = df.dropna(subset=["checkin_date", "user_id"])
    df["trackable_value"] = pd.to_numeric(df["trackable_value"], errors="coerce")
    print(f"  {len(df)} rows, {df['user_id'].nunique()} users")

    # Filter to users with enough daily checkins.
    user_days = df.groupby("user_id")["checkin_date"].nunique()
    valid_users = user_days[user_days >= min_days].index
    print(f"  {len(valid_users)} users with >= {min_days} days")

    if len(valid_users) > max_subjects:
        # Take top N by number of days (most data-rich).
        top = user_days.loc[valid_users].nlargest(max_subjects).index
        valid_users = top
        print(f"  Capped to top {max_subjects} by day count")

    df = df[df["user_id"].isin(valid_users)]

    # Assign integer subject_ids.
    uid_map = {uid: i for i, uid in enumerate(sorted(df["user_id"].unique()))}
    df["subject_id"] = df["user_id"].map(uid_map)

    # Pivot symptoms to wide format (one column per symptom, value 0-4).
    symptoms = df[df["trackable_type"] == "Symptom"].copy()
    sym_pivot = symptoms[symptoms["trackable_name"].isin(TOP_SYMPTOMS)].pivot_table(
        index=["subject_id", "checkin_date"],
        columns="trackable_name",
        values="trackable_value",
        aggfunc="max",  # if multiple entries per day, take worst
    )
    sym_pivot.columns = [f"sym_{c.lower().replace(' ', '_')}" for c in sym_pivot.columns]

    # Pivot weather features.
    weather = df[df["trackable_type"] == "Weather"].copy()
    weather_pivot = weather[weather["trackable_name"].isin(WEATHER_FEATURES)].pivot_table(
        index=["subject_id", "checkin_date"],
        columns="trackable_name",
        values="trackable_value",
        aggfunc="mean",
    )
    weather_pivot.columns = [f"wx_{c}" for c in weather_pivot.columns]

    # Treatment events (binary: any treatment taken that day).
    treatments = df[df["trackable_type"] == "Treatment"].copy()
    treat_days = treatments.groupby(["subject_id", "checkin_date"]).size().reset_index(name="n_treatments")
    treat_days["has_treatment"] = 1

    # Tags as binary indicators for common ones.
    tags = df[df["trackable_type"] == "Tag"].copy()
    top_tags = ["tired", "stressed", "good sleep", "period"]
    tag_pivot = tags[tags["trackable_name"].isin(top_tags)].pivot_table(
        index=["subject_id", "checkin_date"],
        columns="trackable_name",
        values="trackable_value",
        aggfunc="max",
    )
    tag_pivot.columns = [f"tag_{c.replace(' ', '_')}" for c in tag_pivot.columns]
    # Tags are often just presence (value=1) or severity.
    tag_pivot = (tag_pivot > 0).astype(float)

    # Condition severity (overall daily condition score, 0-4).
    conditions = df[df["trackable_type"] == "Condition"].copy()
    cond_daily = conditions.groupby(["subject_id", "checkin_date"])["trackable_value"].max().reset_index()
    cond_daily = cond_daily.rename(columns={"trackable_value": "condition_severity"})

    # Merge everything on (subject_id, checkin_date).
    # Start with all unique (subject, date) pairs.
    all_dates = df[["subject_id", "checkin_date"]].drop_duplicates()
    visits = all_dates.copy()
    visits = visits.merge(sym_pivot, on=["subject_id", "checkin_date"], how="left")
    visits = visits.merge(weather_pivot, on=["subject_id", "checkin_date"], how="left")
    visits = visits.merge(tag_pivot, on=["subject_id", "checkin_date"], how="left")
    visits = visits.merge(
        treat_days[["subject_id", "checkin_date", "has_treatment"]],
        on=["subject_id", "checkin_date"], how="left",
    )
    visits = visits.merge(cond_daily, on=["subject_id", "checkin_date"], how="left")

    visits = visits.sort_values(["subject_id", "checkin_date"]).reset_index(drop=True)

    # Day offset per subject.
    visits["day"] = visits.groupby("subject_id")["checkin_date"].transform(
        lambda x: (x - x.min()).dt.days
    )
    visits["visit_id"] = np.arange(len(visits))

    # Dysregulation score = mean of symptom severities (0-4 normalized to 0-1).
    sym_cols = [c for c in visits.columns if c.startswith("sym_")]
    if sym_cols:
        visits["global_dysregulation_score"] = visits[sym_cols].mean(axis=1) / 4.0
    else:
        visits["global_dysregulation_score"] = np.nan

    # Next-day targets.
    visits["next_day_score"] = visits.groupby("subject_id")["global_dysregulation_score"].shift(-1)
    med = visits["global_dysregulation_score"].median()
    visits["flare_next_day"] = (visits["next_day_score"] > med).astype(float)
    visits.loc[visits["next_day_score"].isna(), "flare_next_day"] = np.nan

    # Treatment events for events.csv.
    events_rows = []
    treat_mask = visits["has_treatment"] == 1
    for _, row in visits[treat_mask].iterrows():
        events_rows.append({
            "subject_id": int(row["subject_id"]),
            "day": int(row["day"]),
            "event_type": "treatment_event",
        })

    # Subjects.
    subjects_df = df.groupby("subject_id").agg(
        age=("age", "first"),
        sex=("sex", "first"),
        country=("country", "first"),
    ).reset_index()

    # Write.
    drop_cols = ["checkin_date", "has_treatment"]
    visits.drop(columns=[c for c in drop_cols if c in visits.columns]).to_csv(
        out / "visits.csv", index=False
    )
    subjects_df.to_csv(out / "subjects.csv", index=False)
    if events_rows:
        pd.DataFrame(events_rows).to_csv(out / "events.csv", index=False)

    obs_cols = [c for c in visits.columns if c.startswith(("sym_", "wx_", "tag_"))
                or c in ("condition_severity",)]
    print(f"\nFlaredown adapter complete.")
    print(f"  Subjects: {visits['subject_id'].nunique()}")
    print(f"  Visits:   {len(visits)}")
    print(f"  Features: {len(obs_cols)} ({obs_cols[:8]}...)")
    print(f"  Dysreg coverage: {visits['global_dysregulation_score'].notna().mean():.1%}")
    print(f"  Events: {len(events_rows)} treatment days")
    print(f"  Output: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Flaredown to GRM-TCM format.")
    parser.add_argument("--input", default="flares-export.csv")
    parser.add_argument("--output-dir", default="flaredown_grm_tcm")
    parser.add_argument("--min-days", type=int, default=60)
    parser.add_argument("--max-subjects", type=int, default=300)
    args = parser.parse_args()
    convert_flaredown(args.input, args.output_dir, args.min_days, args.max_subjects)
