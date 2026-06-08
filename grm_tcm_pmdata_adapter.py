"""PMData dataset adapter for the GRM-TCM pipeline.

Converts the PMData sports-logging dataset (Simula) into visits.csv /
subjects.csv / events.csv format expected by grm_tcm_train.py.

PMData: 16 participants × ~150 days, Fitbit Versa 2 + daily wellness +
training load.  Direct download (no login):
  https://datasets.simula.no/downloads/pmdata.zip

Usage:
  # 1. Download and extract needed files:
  curl -L -o pmdata.zip https://datasets.simula.no/downloads/pmdata.zip
  unzip pmdata.zip "*/pmsys/*" "*/fitbit/sleep_score.csv" \\
    "*/fitbit/resting_heart_rate.json" "*/fitbit/steps.json" \\
    "*/fitbit/sedentary_minutes.json" "*/fitbit/lightly_active_minutes.json" \\
    "*/fitbit/moderately_active_minutes.json" "*/fitbit/very_active_minutes.json" \\
    -d pmdata_raw

  # 2. Convert:
  python grm_tcm_pmdata_adapter.py --input-dir pmdata_raw --output-dir pmdata_grm_tcm

  # 3. Run pipeline:
  python grm_tcm_train.py --input-dir pmdata_grm_tcm --graph-feature-source takens --n-modes 8 --rho 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _load_fitbit_json_daily(path: Path, value_col: str) -> Optional[pd.DataFrame]:
    """Load a Fitbit JSON file and aggregate to daily values."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if not data:
        return None

    rows = []
    for entry in data:
        dt = entry.get("dateTime", "")
        val = entry.get("value")
        if isinstance(val, dict):
            val = val.get("value")
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        rows.append({"date": dt[:10], value_col: val})

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # Aggregate to daily (steps are per-minute, others are already daily).
    return df.groupby("date")[[value_col]].sum().reset_index()


def _load_fitbit_json_daily_mean(path: Path, value_col: str) -> Optional[pd.DataFrame]:
    """Like _load_fitbit_json_daily but takes daily mean (for HR-like data)."""
    df = _load_fitbit_json_daily(path, value_col)
    if df is None:
        return None
    # If it was already one-per-day, sum=mean. If multi-per-day, re-aggregate.
    return df


def _load_participant(pdir: Path, subject_id: int) -> Optional[pd.DataFrame]:
    """Load all data for one participant into daily rows."""

    fitbit_dir = pdir / "fitbit"
    pmsys_dir = pdir / "pmsys"
    frames: List[pd.DataFrame] = []

    # Fitbit: sleep_score.csv (already daily CSV).
    sleep_path = fitbit_dir / "sleep_score.csv"
    if sleep_path.exists():
        df = pd.read_csv(sleep_path)
        date_col = [c for c in df.columns if "timestamp" in c.lower() or "date" in c.lower()]
        if date_col:
            df = df.rename(columns={date_col[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.normalize()
        keep = ["date"]
        for c, new_name in [
            ("overall_score", "sleep_score"),
            ("composition_score", "sleep_composition"),
            ("revitalization_score", "sleep_revitalization"),
            ("duration_score", "sleep_duration_score"),
            ("deep_sleep_in_minutes", "deep_sleep_min"),
            ("resting_heart_rate", "sleep_resting_hr"),
            ("restlessness", "restlessness"),
        ]:
            if c in df.columns:
                df = df.rename(columns={c: new_name})
                keep.append(new_name)
        frames.append(df[keep])

    # Fitbit: JSON daily files.
    for fname, col_name in [
        ("resting_heart_rate.json", "resting_hr"),
        ("steps.json", "daily_steps"),
        ("sedentary_minutes.json", "sedentary_min"),
        ("lightly_active_minutes.json", "lightly_active_min"),
        ("moderately_active_minutes.json", "moderately_active_min"),
        ("very_active_minutes.json", "very_active_min"),
    ]:
        df = _load_fitbit_json_daily(fitbit_dir / fname, col_name)
        if df is not None:
            df["date"] = df["date"].dt.normalize()
            frames.append(df)

    # PMSys: wellness.csv.
    wellness_path = pmsys_dir / "wellness.csv"
    if wellness_path.exists():
        df = pd.read_csv(wellness_path)
        date_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if date_col:
            df = df.rename(columns={date_col[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.normalize()
        # Keep wellness columns.
        keep = ["date"]
        for c in ["fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress",
                   "sleep_duration_h"]:
            if c in df.columns:
                keep.append(c)
        frames.append(df[keep])

    # PMSys: srpe.csv (training sessions).
    srpe_path = pmsys_dir / "srpe.csv"
    srpe_df = None
    if srpe_path.exists():
        df = pd.read_csv(srpe_path)
        date_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if date_col:
            df = df.rename(columns={date_col[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.normalize()
        rpe_col = [c for c in df.columns if "exertion" in c.lower() or "rpe" in c.lower()]
        dur_col = [c for c in df.columns if "duration" in c.lower()]
        if rpe_col and dur_col:
            df["training_load"] = pd.to_numeric(df[rpe_col[0]], errors="coerce") * \
                                  pd.to_numeric(df[dur_col[0]], errors="coerce")
            srpe_df = df.groupby("date")[["training_load"]].sum().reset_index()
            frames.append(srpe_df)

    if not frames:
        return None

    # Merge all on date. Strip timezone to avoid tz-aware/naive mismatch.
    for df in frames:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged["subject_id"] = subject_id

    # Day offset from earliest date.
    date_min = merged["date"].min()
    merged["day"] = (merged["date"] - date_min).dt.days

    return merged


def convert_pmdata(input_dir: str, output_dir: str, min_days: int = 20) -> None:
    """Convert PMData to GRM-TCM pipeline format."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find participant directories.
    pdirs = sorted(input_path.glob("p[0-9]*"))
    if not pdirs:
        pdirs = sorted(input_path.glob("*/p[0-9]*"))
    if not pdirs:
        # Try one more level.
        pdirs = sorted(input_path.glob("*/*/p[0-9]*"))
    if not pdirs:
        raise FileNotFoundError(f"No participant dirs (p01, p02, ...) in {input_path}")

    print(f"Found {len(pdirs)} participant directories")
    all_visits = []
    subjects = []
    all_events = []

    for i, pdir in enumerate(pdirs):
        pid = pdir.name
        print(f"  {pid}...", end=" ")
        df = _load_participant(pdir, subject_id=i)
        if df is None or len(df) < min_days:
            print(f"skip ({0 if df is None else len(df)} days)")
            continue
        obs_cols = [c for c in df.columns if c not in {"date", "subject_id", "day"}
                    and df[c].dtype in [np.float64, np.int64, float, int]]
        print(f"{len(df)} days, {len(obs_cols)} features")
        all_visits.append(df)
        subjects.append({"subject_id": i, "participant": pid})

        # Mark training days as events.
        if "training_load" in df.columns:
            load_days = df[df["training_load"].notna() & (df["training_load"] > 0)]
            for _, row in load_days.iterrows():
                all_events.append({
                    "subject_id": i, "day": int(row["day"]),
                    "event_type": "treatment_event",
                })

    if not all_visits:
        raise ValueError("No valid participants.")

    visits = pd.concat(all_visits, ignore_index=True)
    visits = visits.sort_values(["subject_id", "day"]).reset_index(drop=True)
    visits["visit_id"] = np.arange(len(visits))

    obs_cols = sorted(c for c in visits.columns
                      if c not in {"date", "subject_id", "day", "visit_id"}
                      and visits[c].dtype in [np.float64, np.int64, float, int])

    # Dysregulation score from wellness (higher fatigue/stress/soreness + lower mood/readiness).
    wellness_pos = ["fatigue", "stress", "soreness"]
    wellness_neg = ["mood", "readiness"]
    pos = [c for c in wellness_pos if c in visits.columns]
    neg = [c for c in wellness_neg if c in visits.columns]

    if pos or neg:
        parts = []
        for c in pos:
            cmin, cmax = visits[c].min(), visits[c].max()
            if cmax > cmin:
                parts.append((visits[c] - cmin) / (cmax - cmin))
        for c in neg:
            cmin, cmax = visits[c].min(), visits[c].max()
            if cmax > cmin:
                parts.append(1.0 - (visits[c] - cmin) / (cmax - cmin))
        if parts:
            visits["global_dysregulation_score"] = np.nanmean(
                np.column_stack([s.to_numpy() for s in parts]), axis=1
            )
    if "global_dysregulation_score" not in visits.columns:
        visits["global_dysregulation_score"] = np.nan

    # Next-day targets.
    visits["next_day_score"] = visits.groupby("subject_id")["global_dysregulation_score"].shift(-1)
    med = visits["global_dysregulation_score"].median()
    visits["flare_next_day"] = (visits["next_day_score"] > med).astype(float)
    visits.loc[visits["next_day_score"].isna(), "flare_next_day"] = np.nan

    # Write.
    visits.drop(columns=["date"], errors="ignore").to_csv(output_path / "visits.csv", index=False)
    pd.DataFrame(subjects).to_csv(output_path / "subjects.csv", index=False)
    if all_events:
        pd.DataFrame(all_events).to_csv(output_path / "events.csv", index=False)

    print(f"\nPMData adapter complete.")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Visits:   {len(visits)}")
    print(f"  Features: {len(obs_cols)} ({obs_cols})")
    print(f"  Dysreg coverage: {visits['global_dysregulation_score'].notna().mean():.1%}")
    print(f"  Events: {len(all_events)} training-load days")
    print(f"  Output: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PMData to GRM-TCM format.")
    parser.add_argument("--input-dir", default="pmdata_raw")
    parser.add_argument("--output-dir", default="pmdata_grm_tcm")
    parser.add_argument("--min-days", type=int, default=20)
    args = parser.parse_args()
    convert_pmdata(args.input_dir, args.output_dir, args.min_days)
