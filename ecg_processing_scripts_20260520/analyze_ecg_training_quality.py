"""
ECG training data quality assessment — equivalent to analyze_gsr_training_quality.py.

Combines ECG HRV features and anomaly analysis results to produce
per-subject keep/drop/review recommendations for ML training data.

Reads:
  - ICSlab_project_ECG_output/ECG_HRV_features.csv
  - ICSlab_project_ECG_output/ECG_anomaly_analysis/reports/ecg_anomaly_summary.csv

Outputs:
  - ECG_training_quality_report/ecg_subject_task_quality_detail.csv
  - ECG_training_quality_report/ecg_discard_recommendations.csv
  - ECG_training_quality_report/ecg_review_recommendations.csv
  - ECG_training_quality_report/ecg_subject_level_summary.csv
  - ECG_training_quality_report/ECG_training_data_quality_report.md
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(r"C:\Users\26354\Desktop\saili_data")
ECG_OUTPUT = BASE_DIR / "ICSlab_project_ECG_output"
FEATURES_CSV = ECG_OUTPUT / "ECG_HRV_features.csv"
ANOMALY_SUMMARY = ECG_OUTPUT / "ECG_anomaly_analysis" / "reports" / "ecg_anomaly_summary.csv"

OUT_DIR = BASE_DIR / "ICSlab_project_ECG_output" / "ECG_training_quality_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["T12", "T22", "T32"]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Load features
    if not FEATURES_CSV.exists():
        print(f"[ERROR] Features CSV not found: {FEATURES_CSV}")
        return
    df_features = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    df_features["user_id"] = df_features["user_id"].astype(str).str.strip()
    df_features["task_id"] = df_features["task_id"].astype(str).str.strip()
    print(f"Loaded {len(df_features)} feature rows")

    # Load anomaly summary
    anomaly_map = {}
    if ANOMALY_SUMMARY.exists():
        df_anomaly = pd.read_csv(ANOMALY_SUMMARY, encoding="utf-8-sig")
        for _, row in df_anomaly.iterrows():
            sid = str(row["Subject"]).strip()
            try:
                anomaly_map[sid] = {
                    "total_points": int(row["TotalPoints"]),
                    "serious_fraction": float(row["SeriousAnomalyFraction"]),
                    "all_fraction": float(row["AllAnomalyFraction"]),
                    "t12_serious_frac": float(row.get("T12_SeriousFrac", np.nan) or np.nan),
                    "t22_serious_frac": float(row.get("T22_SeriousFrac", np.nan) or np.nan),
                    "t32_serious_frac": float(row.get("T32_SeriousFrac", np.nan) or np.nan),
                    "t12_all_frac": float(row.get("T12_AllFrac", np.nan) or np.nan),
                    "t22_all_frac": float(row.get("T22_AllFrac", np.nan) or np.nan),
                    "t32_all_frac": float(row.get("T32_AllFrac", np.nan) or np.nan),
                    "flags": str(row.get("Flags", "")),
                }
            except (ValueError, KeyError):
                continue
        print(f"Loaded {len(anomaly_map)} anomaly summaries")
    else:
        print(f"[WARN] Anomaly summary not found: {ANOMALY_SUMMARY}")

    # Get all unique subjects
    all_subjects = sorted(df_features["user_id"].unique(),
                          key=lambda x: int(x.lstrip("T")) if x.lstrip("T").isdigit() else 999)
    expected_trials = len(all_subjects) * 3

    detail_rows = []
    discard_rows = []
    review_rows = []

    for subject in all_subjects:
        subj_features = df_features[df_features["user_id"] == subject]

        for task_id in CONDITIONS:
            row_data = subj_features[subj_features["task_id"] == task_id]
            feature_row = row_data.iloc[0].to_dict() if len(row_data) > 0 else {}

            anom = anomaly_map.get(subject, {})
            serious_fraction = anom.get("serious_fraction", np.nan)
            all_fraction = anom.get("all_fraction", np.nan)

            # Use per-condition fractions if available
            task_serious_key = {"T12": "t12_serious_frac", "T22": "t22_serious_frac", "T32": "t32_serious_frac"}.get(task_id, "")
            task_all_key = {"T12": "t12_all_frac", "T22": "t22_all_frac", "T32": "t32_all_frac"}.get(task_id, "")
            task_serious_frac = anom.get(task_serious_key, np.nan) if task_serious_key else np.nan
            task_all_frac = anom.get(task_all_key, np.nan) if task_all_key else np.nan

            # Use per-condition fractions when available, otherwise fall back to subject-level
            s_frac = task_serious_frac if not np.isnan(task_serious_frac) else serious_fraction
            a_frac = task_all_frac if not np.isnan(task_all_frac) else all_fraction

            # Determine recommendation
            recommendation = "keep"
            reasons = []

            if len(row_data) == 0:
                recommendation = "drop"
                reasons.append("missing_feature_row")
            else:
                # Check feature flags
                flags = str(feature_row.get("quality_flags", ""))
                n_peaks = feature_row.get("n_peaks", 0)
                peak_rate = feature_row.get("peak_rate_per_min", np.nan)
                duration = feature_row.get("duration_sec", 0)
                ecg_std = feature_row.get("ecg_std", 0)

                if "too_short" in flags:
                    recommendation = "drop"
                    reasons.append("too_short_duration")

                if "flatline_signal" in flags:
                    recommendation = "drop"
                    reasons.append("flatline_signal")

                if "too_few_peaks" in flags:
                    recommendation = "drop"
                    reasons.append("too_few_peaks")

                if duration < 60:
                    if recommendation == "keep":
                        recommendation = "review"
                    reasons.append("short_duration_(<60s)")

                if ecg_std > 1.0:
                    if recommendation == "keep":
                        recommendation = "review"
                    reasons.append("high_variation")

                if n_peaks > 0 and (peak_rate < 30 or peak_rate > 200):
                    if recommendation == "keep":
                        recommendation = "review"
                    reasons.append("abnormal_peak_rate")

                # Two-tier anomaly-based rules (mirrors GSR)
                if not np.isnan(s_frac):
                    if s_frac >= 0.01:  # serious >= 1%
                        recommendation = "drop"
                        reasons.append("high_serious_anomaly_fraction_(>=1%)")
                    elif s_frac >= 0.001:  # serious >= 0.1%
                        if recommendation == "keep":
                            recommendation = "review"
                        reasons.append("elevated_serious_anomaly_fraction_(>=0.1%)")

                if not np.isnan(a_frac):
                    if a_frac >= 0.10:  # all anomalies >= 10%
                        recommendation = "drop"
                        reasons.append("high_all_anomaly_fraction_(>=10%)")
                    elif a_frac >= 0.02:  # all anomalies >= 2%
                        if recommendation == "keep":
                            recommendation = "review"
                        reasons.append("elevated_all_anomaly_fraction_(>=2%)")

            detail_rows.append({
                "user_id": subject,
                "task_id": task_id,
                "condition": feature_row.get("condition", ""),
                "recommendation": recommendation,
                "reasons": ";".join(reasons),
                "n_samples": feature_row.get("n_samples", ""),
                "duration_sec": feature_row.get("duration_sec", ""),
                "ecg_mean": feature_row.get("ecg_mean", ""),
                "ecg_std": feature_row.get("ecg_std", ""),
                "ecg_min": feature_row.get("ecg_min", ""),
                "ecg_max": feature_row.get("ecg_max", ""),
                "n_peaks": feature_row.get("n_peaks", ""),
                "peak_rate_per_min": feature_row.get("peak_rate_per_min", ""),
                "mean_ibi_ms": feature_row.get("mean_ibi_ms", ""),
                "cv_ibi_pct": feature_row.get("cv_ibi_pct", ""),
                "lf_hf_ratio": feature_row.get("lf_hf_ratio", ""),
                "dominant_freq_hz": feature_row.get("dominant_freq_hz", ""),
                "quality_flags": feature_row.get("quality_flags", ""),
                "serious_anomaly_fraction": f"{s_frac:.6f}" if not np.isnan(s_frac) else "",
                "all_anomaly_fraction": f"{a_frac:.6f}" if not np.isnan(a_frac) else "",
                "anomaly_flags": anom.get("flags", ""),
            })

            if recommendation == "drop":
                discard_rows.append(detail_rows[-1])
            elif recommendation == "review":
                review_rows.append(detail_rows[-1])

    # Save detail CSV
    detail_fields = [
        "user_id", "task_id", "condition", "recommendation", "reasons",
        "n_samples", "duration_sec", "ecg_mean", "ecg_std", "ecg_min", "ecg_max",
        "n_peaks", "peak_rate_per_min", "mean_ibi_ms", "cv_ibi_pct",
        "lf_hf_ratio", "dominant_freq_hz", "quality_flags",
        "serious_anomaly_fraction", "all_anomaly_fraction", "anomaly_flags",
    ]
    write_csv(OUT_DIR / "ecg_subject_task_quality_detail.csv", detail_fields, detail_rows)
    write_csv(OUT_DIR / "ecg_discard_recommendations.csv", detail_fields, discard_rows)
    write_csv(OUT_DIR / "ecg_review_recommendations.csv", detail_fields, review_rows)

    # Subject-level summary
    subject_summary = []
    for subject in all_subjects:
        subj_rows = [r for r in detail_rows if r["user_id"] == subject]
        keep = sum(1 for r in subj_rows if r["recommendation"] == "keep")
        review = sum(1 for r in subj_rows if r["recommendation"] == "review")
        drop = sum(1 for r in subj_rows if r["recommendation"] == "drop")
        subject_summary.append({
            "user_id": subject,
            "keep_count": str(keep),
            "review_count": str(review),
            "drop_count": str(drop),
            "status": "ok" if drop == 0 and review == 0 else ("drop" if drop >= 2 else "review"),
        })

    write_csv(
        OUT_DIR / "ecg_subject_level_summary.csv",
        ["user_id", "keep_count", "review_count", "drop_count", "status"],
        subject_summary,
    )

    # Statistics
    n_keep = sum(1 for r in detail_rows if r["recommendation"] == "keep")
    n_review = sum(1 for r in detail_rows if r["recommendation"] == "review")
    n_drop = sum(1 for r in detail_rows if r["recommendation"] == "drop")
    n_total = len(detail_rows)
    n_subjects_ok = sum(1 for s in subject_summary if s["status"] == "ok")
    n_subjects_drop = sum(1 for s in subject_summary if s["status"] == "drop")

    # Markdown report
    report = f"""# ECG Training Data Quality Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

| Metric | Value |
|--------|-------|
| Total subjects | {len(all_subjects)} |
| Expected trials (3 per subject) | {expected_trials} |
| Actual feature rows | {n_total} |
| **Keep** | {n_keep} ({100*n_keep/max(1,n_total):.1f}%) |
| **Review** | {n_review} ({100*n_review/max(1,n_total):.1f}%) |
| **Drop** | {n_drop} ({100*n_drop/max(1,n_total):.1f}%) |
| Subjects with all 3 tasks clean | {n_subjects_ok} |
| Subjects with >=2 tasks dropped | {n_subjects_drop} |

## Drop Reasons

| Reason | Count |
|--------|-------|
"""
    # Count drop reasons
    drop_reason_counts = {}
    for r in discard_rows:
        for reason in r["reasons"].split(";"):
            reason = reason.strip()
            if reason:
                drop_reason_counts[reason] = drop_reason_counts.get(reason, 0) + 1

    for reason, count in sorted(drop_reason_counts.items(), key=lambda x: -x[1]):
        report += f"| {reason} | {count} |\n"

    report += f"""
## Recommendations by Subject

| Subject | Keep | Review | Drop | Status |
|---------|------|--------|------|--------|
"""
    for s in subject_summary:
        report += f"| {s['user_id']} | {s['keep_count']} | {s['review_count']} | {s['drop_count']} | {s['status']} |\n"

    report += f"""
## Files

| File | Path |
|------|------|
| Detail CSV | {OUT_DIR / 'ecg_subject_task_quality_detail.csv'} |
| Discard list | {OUT_DIR / 'ecg_discard_recommendations.csv'} |
| Review list | {OUT_DIR / 'ecg_review_recommendations.csv'} |
| Subject summary | {OUT_DIR / 'ecg_subject_level_summary.csv'} |
"""
    (OUT_DIR / "ECG_training_data_quality_report.md").write_text(report, encoding="utf-8")

    # Print summary
    print(f"\n=== ECG Training Data Quality Report ===")
    print(f"Total: {n_total} trials ({len(all_subjects)} subjects)")
    print(f"  Keep:   {n_keep:4d} ({100*n_keep/max(1,n_total):5.1f}%)")
    print(f"  Review: {n_review:4d} ({100*n_review/max(1,n_total):5.1f}%)")
    print(f"  Drop:   {n_drop:4d} ({100*n_drop/max(1,n_total):5.1f}%)")
    print(f"Clean subjects (all 3 tasks keep): {n_subjects_ok}")
    print(f"Dropped subjects (>=2 tasks): {n_subjects_drop}")
    print(f"\nTop drop reasons:")
    for reason, count in sorted(drop_reason_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {reason}: {count}")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
