"""
ECG anomaly detection — equivalent to analyze_gsr_anomalies.py.

Reads processed ECG CSV files (T12+T22+T32 combined per subject),
detects anomalies, generates per-subject plots with anomalies marked in red,
overview plots, and an interactive HTML report.

Anomaly rules adapted for ECG sensor data:
  Serious (≥1% → drop in quality assessment):
    - non_finite: NaN or Inf values
    - extreme_amplitude: |value| > 2.0 (beyond typical range for this sensor)
    - large_step: adjacent difference > 1.0 (implausible jump for 200Hz interpolated data)
  Minor (only counted in "all" ≥10% → drop):
    - flatline_window: constant value over a >= 2-second window
    - robust_outlier_mad_gt_8: Modified Z-score > 8 using median absolute deviation
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(r"C:\Users\26354\Desktop\saili_data")
OUT_DIR = BASE_DIR / "ICSlab_project_ECG_output" / "ECG_anomaly_analysis"
PLOTS_DIR = OUT_DIR / "per_subject_plots"
OVERVIEW_DIR = OUT_DIR / "overview"
REPORTS_DIR = OUT_DIR / "reports"

for folder in (PLOTS_DIR, OVERVIEW_DIR, REPORTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# --- Two-tier anomaly classification (mirrors GSR) ---
# Serious: count toward ≥1% threshold → drop
SERIOUS_TYPES = {"non_finite", "extreme_amplitude_gt_2", "large_step_gt_1"}
# Minor: only counted in "all" ≥10% threshold
MINOR_TYPES = {"flatline_window", "robust_outlier_mad_gt_8"}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_subject_combined(subject_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Read T12+T22+T32 processed ECG CSV files for a subject,
    concatenate them into a single time series.
    Returns (values, segment_starts, segment_labels, parse_errors).
    """
    ecg_dir = BASE_DIR / subject_id / "processed_signal_data" / "ECG"
    if not ecg_dir.is_dir():
        return np.array([]), np.array([], dtype=int), np.array([], dtype=str), []

    all_values = []
    all_labels = []
    segment_starts = []
    parse_errors = []

    for label, tag in [("T12", "low"), ("T22", "medium"), ("T32", "high")]:
        csv_path = ecg_dir / f"{subject_id}_ECG_{label}_{tag}_trust.csv"
        if not csv_path.exists():
            continue

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            start_idx = len(all_values)
            segment_starts.append(start_idx)
            for row_num, row in enumerate(reader, start=2):
                try:
                    all_values.append(float(row["Data"]))
                    all_labels.append(tag)
                except Exception as exc:
                    parse_errors.append({
                        "Subject": subject_id,
                        "File": str(csv_path),
                        "Row": str(row_num),
                        "Timestamp": row.get("Timestamp", ""),
                        "Data": row.get("Data", ""),
                        "Reason": f"parse_error:{exc}",
                    })

    values = np.array(all_values, dtype=float)
    starts = np.array(segment_starts, dtype=int)
    labels = np.array(all_labels, dtype=str)

    return values, starts, labels, parse_errors


def median_abs_deviation(values: np.ndarray, med: float) -> float:
    return float(np.median(np.abs(values - med)))


def detect_anomalies(values: np.ndarray) -> tuple[list[set[str]], dict]:
    """Detect anomalies in the ECG signal."""
    n = len(values)
    reasons = [set() for _ in range(n)]
    finite = np.isfinite(values)

    # Rule 1: Non-finite values
    for idx in np.where(~finite)[0]:
        reasons[idx].add("non_finite")

    if n == 0 or not np.any(finite):
        return reasons, {}

    finite_vals = values[finite]

    # Rule 2: Extreme amplitude
    extreme = finite & (np.abs(values) > 2.0)
    for idx in np.where(extreme)[0]:
        reasons[idx].add("extreme_amplitude_gt_2")

    # Rule 3: Flatline detection (rolling window std)
    window_samples = int(200 * 2.0)  # 2 seconds at 200 Hz
    if window_samples >= 4 and n >= window_samples:
        # Use convolution for efficient rolling std
        for start in range(0, n - window_samples + 1, window_samples // 2):
            end = min(start + window_samples, n)
            segment = values[start:end]
            seg_finite = np.isfinite(segment)
            if np.sum(seg_finite) < 2:
                continue
            seg_std = float(np.std(segment[seg_finite]))
            if seg_std < 1e-6:
                for idx in range(start, end):
                    if idx < n and np.isfinite(values[idx]):
                        reasons[idx].add("flatline_window")

    # Rule 4: Large step
    diffs = np.diff(values)
    large_step_mask = np.isfinite(diffs) & (np.abs(diffs) > 1.0)
    for idx in np.where(large_step_mask)[0] + 1:
        reasons[idx].add("large_step_gt_1")

    # Rule 5: Robust MAD outlier
    valid_for_robust = finite & (np.abs(values) <= 2.0)
    robust_vals = values[valid_for_robust]
    if len(robust_vals) >= 20:
        med = float(np.median(robust_vals))
        mad = median_abs_deviation(robust_vals, med)
        if mad > 0:
            modified_z = np.zeros(n, dtype=float)
            modified_z[valid_for_robust] = (
                0.6745 * np.abs(values[valid_for_robust] - med) / mad
            )
            for idx in np.where(modified_z > 8)[0]:
                reasons[idx].add("robust_outlier_mad_gt_8")
    else:
        med = float(np.median(finite_vals))
        mad = 0.0

    # Statistics: separate serious vs all
    counts = Counter()
    serious_counts = Counter()
    for reason_set in reasons:
        for reason in reason_set:
            counts[reason] += 1
            if reason in SERIOUS_TYPES:
                serious_counts[reason] += 1

    # Also count parse_errors as serious anomalies (mirrors GSR treatment)
    n_serious = sum(serious_counts.values())
    n_all = sum(counts.values())

    mode_value, mode_count = Counter(finite_vals.round(6).tolist()).most_common(1)[0] if len(finite_vals) > 0 else (np.nan, 0)

    stats = {
        "min": float(np.min(finite_vals)) if len(finite_vals) > 0 else np.nan,
        "p01": float(np.percentile(finite_vals, 1)) if len(finite_vals) > 0 else np.nan,
        "median": float(np.median(finite_vals)) if len(finite_vals) > 0 else np.nan,
        "p99": float(np.percentile(finite_vals, 99)) if len(finite_vals) > 0 else np.nan,
        "max": float(np.max(finite_vals)) if len(finite_vals) > 0 else np.nan,
        "mode_value": float(mode_value),
        "mode_fraction": float(mode_count / len(finite_vals)) if len(finite_vals) > 0 else np.nan,
        "extreme_fraction": float(np.sum(extreme) / n) if n > 0 else np.nan,
        "reason_counts": counts,
        "serious_reason_counts": serious_counts,
        "n_serious": n_serious,
        "n_all": n_all,
    }
    return reasons, stats


def plot_subject(subject_id: str, values: np.ndarray, reasons: list[set[str]],
                 stats: dict, segment_starts: np.ndarray, segment_labels: np.ndarray) -> Path:
    """Generate per-subject ECG anomaly plot (combined T12+T22+T32)."""
    png = PLOTS_DIR / f"{subject_id}_ECG.png"
    n = len(values)

    if n == 0:
        fig, ax = plt.subplots(figsize=(10, 3), dpi=120)
        ax.text(0.5, 0.5, f"No ECG data for {subject_id}", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        return png

    x_minutes = np.arange(n) / 200.0 / 60.0  # minutes
    anomaly_mask = np.array([bool(r) for r in reasons], dtype=bool)

    fig, ax = plt.subplots(figsize=(14, 4), dpi=120)

    # Plot segments separately with different colors
    colors = {"low": "#2196f3", "medium": "#ff9800", "high": "#4caf50"}
    if len(segment_starts) > 0:
        for i, start in enumerate(segment_starts):
            end = segment_starts[i + 1] if i + 1 < len(segment_starts) else n
            label = segment_labels[start] if start < n else "unknown"
            color = colors.get(label, "#9e9e9e")
            ax.plot(x_minutes[start:end], values[start:end],
                   color=color, linewidth=0.35, alpha=0.8)
    else:
        ax.plot(x_minutes, values, color="#52616b", linewidth=0.35, alpha=0.8)

    if np.any(anomaly_mask):
        ax.scatter(
            x_minutes[anomaly_mask], values[anomaly_mask],
            s=5, color="#d7191c", alpha=0.7, label=f"anomaly ({int(np.sum(anomaly_mask))})",
            zorder=3,
        )
        ax.legend(loc="upper right", fontsize=7, frameon=False)

    n_anomaly = int(np.sum(anomaly_mask))
    ax.set_title(
        f"{subject_id} ECG (combined T12+T22+T32) | n={n} | anomalies={n_anomaly} ({100*n_anomaly/n:.1f}%) | "
        f"min={stats.get('min', np.nan):.4f}, med={stats.get('median', np.nan):.4f}, max={stats.get('max', np.nan):.4f}",
        fontsize=9,
    )
    ax.set_xlabel("Minutes from first processed ECG sample")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linewidth=0.3, alpha=0.35)

    # Add trust condition labels
    if len(segment_starts) > 0:
        y_max = np.nanmax(values[np.isfinite(values)]) if np.any(np.isfinite(values)) else 1.0
        for i, start in enumerate(segment_starts):
            end = segment_starts[i + 1] if i + 1 < len(segment_starts) else n
            mid = (x_minutes[start] + x_minutes[end - 1]) / 2
            label = segment_labels[start] if start < n else "?"
            ax.annotate(label, xy=(mid, y_max), xytext=(0, 8),
                       textcoords="offset points", ha="center", fontsize=7,
                       color=colors.get(label, "#333333"),
                       arrowprops=dict(arrowstyle="->", color=colors.get(label, "#333333"), lw=0.5))

    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_overview(summary_rows: list[dict], all_finite_values: np.ndarray) -> None:
    """Generate overview plots."""
    # Top anomaly subjects bar chart
    if summary_rows:
        top = sorted(summary_rows, key=lambda r: int(r["AllAnomalyPoints"]), reverse=True)[:30]
        labels = [r["Subject"] for r in top]
        counts = [int(r["AllAnomalyPoints"]) for r in top]
        fractions = [float(r["AllAnomalyFraction"]) * 100 for r in top]

        fig, ax1 = plt.subplots(figsize=(12, 5), dpi=120)
        bars = ax1.bar(labels, counts, color="#d7191c", alpha=0.85)
        ax1.set_title("Top 30 subjects by anomaly count (ECG)")
        ax1.set_xlabel("Subject")
        ax1.set_ylabel("Anomaly points")
        ax1.tick_params(axis="x", rotation=70)
        ax1.grid(axis="y", linewidth=0.3, alpha=0.35)

        ax2 = ax1.twinx()
        ax2.plot(labels, fractions, color="#2196f3", marker="o", markersize=4, linewidth=1.0)
        ax2.set_ylabel("Anomaly fraction (%)", color="#2196f3")
        ax2.tick_params(axis="y", labelcolor="#2196f3")

        fig.tight_layout()
        fig.savefig(OVERVIEW_DIR / "top_anomaly_subjects.png")
        plt.close(fig)

    # Global value distribution
    if len(all_finite_values) > 0:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        ax.hist(all_finite_values, bins=200, color="#52616b", alpha=0.85, edgecolor="#263238", linewidth=0.2)
        ax.set_yscale("log")
        ax.set_title(f"ECG value distribution across all subjects (n={len(all_finite_values):,})")
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Count (log scale)")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        fig.tight_layout()
        fig.savefig(OVERVIEW_DIR / "global_value_distribution.png")
        plt.close(fig)

    # Anomaly reason distribution
    if summary_rows:
        reason_totals = Counter()
        for row in summary_rows:
            for reason, count_str in [
                ("non_finite", row.get("NonFinite", "0")),
                ("flatline_window", row.get("Flatline", "0")),
                ("extreme_amplitude", row.get("ExtremeAmp", "0")),
                ("large_step", row.get("LargeStep", "0")),
                ("robust_outlier", row.get("RobustOutlier", "0")),
            ]:
                try:
                    reason_totals[reason] += int(count_str)
                except (ValueError, TypeError):
                    pass

        if reason_totals:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
            reasons_sorted = reason_totals.most_common()
            labels = [r[0] for r in reasons_sorted]
            counts = [r[1] for r in reasons_sorted]
            colors = ["#d7191c", "#fdae61", "#a6d96a", "#1a9641", "#4575b4"]
            ax.barh(labels, counts, color=colors[:len(labels)])
            ax.set_title("Anomaly reason distribution (ECG)")
            ax.set_xlabel("Count")
            for i, v in enumerate(counts):
                ax.text(v + max(counts) * 0.01, i, str(v), va="center", fontsize=8)
            ax.grid(axis="x", linewidth=0.3, alpha=0.35)
            fig.tight_layout()
            fig.savefig(OVERVIEW_DIR / "anomaly_reason_distribution.png")
            plt.close(fig)


def build_index(summary_rows: list[dict]) -> None:
    """Build interactive HTML report."""
    rows_html = []
    for row in sorted(summary_rows, key=lambda r: int(r["AllAnomalyPoints"]), reverse=True):
        sid = row["Subject"]
        plot_rel = f"per_subject_plots/{sid}_ECG.png"
        rows_html.append(
            "<tr>"
            f"<td>{escape(sid)}</td>"
            f"<td>{escape(row['TotalPoints'])}</td>"
            f"<td>{escape(row['SeriousAnomalyPoints'])}</td>"
            f"<td>{escape(row['SeriousAnomalyFraction'])}</td>"
            f"<td>{escape(row['AllAnomalyPoints'])}</td>"
            f"<td>{escape(row['AllAnomalyFraction'])}</td>"
            f"<td>{escape(row['Min'])}</td>"
            f"<td>{escape(row['Median'])}</td>"
            f"<td>{escape(row['Max'])}</td>"
            f"<td>{escape(row['Flags'])}</td>"
            f"<td><a href='{plot_rel}'><img src='{plot_rel}' width='300'></a></td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>ECG Anomaly Analysis</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; vertical-align: top; }}
th {{ background: #f2f2f2; position: sticky; top: 0; }}
img {{ display: block; max-width: 300px; }}
</style>
</head>
<body>
<h1>ECG Anomaly Analysis</h1>
<p>Rules: non-finite, extreme amplitude (&gt;2), flatline window (2s),
large step (&gt;1), robust MAD outlier (&gt;8).</p>
<p>
<a href="overview/top_anomaly_subjects.png">Top anomaly subjects</a> |
<a href="overview/global_value_distribution.png">Global distribution</a> |
<a href="overview/anomaly_reason_distribution.png">Reason distribution</a> |
<a href="reports/ecg_anomaly_summary.csv">Summary CSV</a> |
<a href="reports/ecg_anomaly_points.csv">Anomaly points CSV</a>
</p>
<table>
<thead>
<tr><th>Subject</th><th>Points</th><th>Anomalies</th><th>Fraction</th>
<th>Min</th><th>Median</th><th>Max</th><th>Flags</th><th>Plot</th></tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    summary_rows = []
    anomaly_rows = []
    all_finite_values = []

    for n in range(1, 159):
        subject_id = f"T{n:03d}"
        values, segment_starts, segment_labels, parse_errors = read_subject_combined(subject_id)

        if len(values) == 0:
            continue

        reasons, stats = detect_anomalies(values)
        plot_subject(subject_id, values, reasons, stats, segment_starts, segment_labels)

        # Collect finite values for global histogram
        finite_vals = values[np.isfinite(values)]
        if len(finite_vals) > 0:
            all_finite_values.append(finite_vals)

        anomaly_mask = np.array([bool(r) for r in reasons], dtype=bool)
        reason_counts = stats.get("reason_counts", Counter())
        n_all = stats.get("n_all", int(np.sum(anomaly_mask)))
        n_serious = stats.get("n_serious", 0)
        n_total = len(values)

        flags = []
        if stats.get("extreme_fraction", 0) >= 0.05:
            flags.append("many_extreme_amplitude")
        if stats.get("mode_fraction", 0) >= 0.8:
            flags.append(f"flatline_mode_{stats.get('mode_value', np.nan):.4f}")
        if n_total == 0:
            flags.append("empty_data")

        # Compute per-condition fractions
        t12_serious = 0
        t12_all = 0
        t12_n = 0
        t22_serious = 0
        t22_all = 0
        t22_n = 0
        t32_serious = 0
        t32_all = 0
        t32_n = 0

        if len(segment_starts) > 0:
            for i, start in enumerate(segment_starts):
                end = segment_starts[i + 1] if i + 1 < len(segment_starts) else n_total
                if start >= n_total:
                    break
                label = segment_labels[start] if start < n_total else ""
                seg_anomaly_mask = anomaly_mask[start:end]
                seg_reasons = reasons[start:end]
                seg_n = end - start
                seg_all = int(np.sum(seg_anomaly_mask))
                seg_serious = sum(
                    1 for r_set in seg_reasons
                    if r_set & SERIOUS_TYPES
                )
                if label == "low":  # T12
                    t12_serious += seg_serious
                    t12_all += seg_all
                    t12_n += seg_n
                elif label == "medium":  # T22
                    t22_serious += seg_serious
                    t22_all += seg_all
                    t22_n += seg_n
                elif label == "high":  # T32
                    t32_serious += seg_serious
                    t32_all += seg_all
                    t32_n += seg_n

        # Count per-reason anomalies
        nonfinite_count = reason_counts.get("non_finite", 0)
        flatline_count = reason_counts.get("flatline_window", 0)
        extreme_count = reason_counts.get("extreme_amplitude_gt_2", 0)
        largestep_count = reason_counts.get("large_step_gt_1", 0)
        robust_count = reason_counts.get("robust_outlier_mad_gt_8", 0)

        summary_rows.append({
            "Subject": subject_id,
            "TotalPoints": str(n_total),
            "SeriousAnomalyPoints": str(n_serious),
            "SeriousAnomalyFraction": f"{(n_serious / n_total):.6f}" if n_total else "",
            "AllAnomalyPoints": str(n_all),
            "AllAnomalyFraction": f"{(n_all / n_total):.6f}" if n_total else "",
            # Per-condition serious fractions (for task-level quality assessment)
            "T12_SeriousFrac": f"{(t12_serious / t12_n):.6f}" if t12_n else "",
            "T22_SeriousFrac": f"{(t22_serious / t22_n):.6f}" if t22_n else "",
            "T32_SeriousFrac": f"{(t32_serious / t32_n):.6f}" if t32_n else "",
            "T12_AllFrac": f"{(t12_all / t12_n):.6f}" if t12_n else "",
            "T22_AllFrac": f"{(t22_all / t22_n):.6f}" if t22_n else "",
            "T32_AllFrac": f"{(t32_all / t32_n):.6f}" if t32_n else "",
            "Min": f"{stats.get('min', np.nan):.6g}",
            "P01": f"{stats.get('p01', np.nan):.6g}",
            "Median": f"{stats.get('median', np.nan):.6g}",
            "P99": f"{stats.get('p99', np.nan):.6g}",
            "Max": f"{stats.get('max', np.nan):.6g}",
            "ModeValue": f"{stats.get('mode_value', np.nan):.6g}",
            "ModeFraction": f"{stats.get('mode_fraction', np.nan):.6f}",
            "ExtremeFraction": f"{stats.get('extreme_fraction', np.nan):.6f}",
            "NonFinite": str(nonfinite_count),
            "Flatline": str(flatline_count),
            "ExtremeAmp": str(extreme_count),
            "LargeStep": str(largestep_count),
            "RobustOutlier": str(robust_count),
            "ParseErrors": str(len(parse_errors)),
            "Flags": "|".join(flags),
        })

        # Detail: each anomaly point
        for idx, reason_set in enumerate(reasons):
            if not reason_set:
                continue
            anomaly_rows.append({
                "Subject": subject_id,
                "Index": str(idx),
                "Data": f"{values[idx]:.12g}",
                "Reasons": "|".join(sorted(reason_set)),
            })

        for pe in parse_errors:
            anomaly_rows.append({
                "Subject": pe["Subject"],
                "Index": pe["Row"],
                "Data": pe["Data"],
                "Reasons": pe["Reason"],
                "SourceFile": pe.get("File", ""),
            })

    # Save CSVs
    write_csv(
        REPORTS_DIR / "ecg_anomaly_summary.csv",
        [
            "Subject", "TotalPoints",
            "SeriousAnomalyPoints", "SeriousAnomalyFraction",
            "AllAnomalyPoints", "AllAnomalyFraction",
            "T12_SeriousFrac", "T22_SeriousFrac", "T32_SeriousFrac",
            "T12_AllFrac", "T22_AllFrac", "T32_AllFrac",
            "Min", "P01", "Median", "P99", "Max",
            "ModeValue", "ModeFraction", "ExtremeFraction",
            "NonFinite", "Flatline", "ExtremeAmp", "LargeStep", "RobustOutlier",
            "ParseErrors", "Flags",
        ],
        summary_rows,
    )
    write_csv(
        REPORTS_DIR / "ecg_anomaly_points.csv",
        ["Subject", "Index", "Data", "Reasons", "SourceFile"],
        anomaly_rows,
    )

    # Overview plots and HTML
    all_vals = np.concatenate(all_finite_values) if all_finite_values else np.array([])
    plot_overview(summary_rows, all_vals)
    build_index(summary_rows)

    # Summary
    total_points = sum(int(r["TotalPoints"]) for r in summary_rows)
    total_serious = sum(int(r["SeriousAnomalyPoints"]) for r in summary_rows)
    total_all = sum(int(r["AllAnomalyPoints"]) for r in summary_rows)
    print(f"\n=== ECG Anomaly Analysis Complete ===")
    print(f"Subjects processed: {len(summary_rows)}")
    print(f"Total data points: {total_points:,}")
    print(f"Serious anomalies: {total_serious:,} ({100*total_serious/total_points:.3f}%)")
    print(f"All anomalies: {total_all:,} ({100*total_all/total_points:.3f}%)")
    print(f"Output: {OUT_DIR}")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  HTML: {OUT_DIR / 'index.html'}")

    print(f"\nTop 20 subjects by all-anomaly count:")
    for row in sorted(summary_rows, key=lambda r: int(r["AllAnomalyPoints"]), reverse=True)[:20]:
        print(f"  {row['Subject']}: serious={row['SeriousAnomalyPoints']} all={row['AllAnomalyPoints']}/{row['TotalPoints']} "
              f"({float(row['AllAnomalyFraction'])*100:.2f}%) flags={row['Flags']}")


if __name__ == "__main__":
    main()
