import csv
import math
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\信任数据集\imag-gsr-ecg")
OUT = ROOT / "GSR_anomaly_analysis_20260509"
PLOTS = OUT / "per_subject_plots"
OVERVIEW = OUT / "overview"
REPORTS = OUT / "reports"

for folder in (PLOTS, OVERVIEW, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)


def parse_ts(value):
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_subject(path):
    timestamps = []
    values = []
    trusts = []
    parse_errors = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            try:
                timestamps.append(parse_ts(row["Timestamp"]))
                values.append(float(row["Data"]))
                trusts.append(row["Trust"])
            except Exception as exc:
                parse_errors.append(
                    {
                        "Row": idx,
                        "Timestamp": row.get("Timestamp", ""),
                        "Data": row.get("Data", ""),
                        "Trust": row.get("Trust", ""),
                        "Reason": f"parse_error:{exc}",
                    }
                )
    return timestamps, np.array(values, dtype=float), trusts, parse_errors


def median_abs_deviation(values, med):
    return float(np.median(np.abs(values - med)))


def detect_anomalies(values):
    reasons = [set() for _ in range(len(values))]
    finite = np.isfinite(values)

    for idx in np.where(~finite)[0]:
        reasons[idx].add("non_finite")

    finite_values = values[finite]
    if len(finite_values) == 0:
        return reasons, {}

    non_positive = finite & (values <= 0)
    sentinel_one = finite & (values == 1)
    high_resistance = finite & (values >= 1000)

    for idx in np.where(non_positive)[0]:
        reasons[idx].add("non_positive_le_0")
    for idx in np.where(sentinel_one)[0]:
        reasons[idx].add("sentinel_value_1")
    for idx in np.where(high_resistance)[0]:
        reasons[idx].add("high_resistance_ge_1000")

    diffs = np.diff(values)
    large_step = np.isfinite(diffs) & (np.abs(diffs) >= 500)
    for idx in np.where(large_step)[0] + 1:
        reasons[idx].add("large_step_ge_500")

    valid_for_robust = finite & (values > 1) & (values < 1000)
    robust_values = values[valid_for_robust]
    if len(robust_values) >= 20:
        med = float(np.median(robust_values))
        mad = median_abs_deviation(robust_values, med)
        if mad > 0:
            modified_z = np.zeros_like(values, dtype=float)
            modified_z[valid_for_robust] = 0.6745 * np.abs(values[valid_for_robust] - med) / mad
            for idx in np.where(modified_z > 8)[0]:
                reasons[idx].add("robust_outlier_mad_gt_8")
    else:
        med = float(np.median(finite_values))
        mad = 0.0

    counts = Counter()
    for reason_set in reasons:
        for reason in reason_set:
            counts[reason] += 1

    mode_value, mode_count = Counter(values[finite].tolist()).most_common(1)[0]
    stats = {
        "min": float(np.min(finite_values)),
        "p01": float(np.percentile(finite_values, 1)),
        "median": float(np.median(finite_values)),
        "p99": float(np.percentile(finite_values, 99)),
        "max": float(np.max(finite_values)),
        "mode_value": float(mode_value),
        "mode_fraction": float(mode_count / len(finite_values)),
        "zero_or_one_fraction": float(np.sum(finite & (values <= 1)) / len(values)),
        "high_ge_1000_fraction": float(np.sum(high_resistance) / len(values)),
        "reason_counts": counts,
    }
    return reasons, stats


def plot_subject(sid, timestamps, values, reasons, stats):
    png = PLOTS / f"{sid}_GSR.png"
    if len(values) == 0:
        fig, ax = plt.subplots(figsize=(10, 3), dpi=120)
        ax.text(0.5, 0.5, "No T12/T22/T32 GSR rows", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        return png

    start = timestamps[0]
    x = np.array([(t - start).total_seconds() / 60 for t in timestamps])
    anomaly_mask = np.array([bool(r) for r in reasons])

    fig, ax = plt.subplots(figsize=(11, 4), dpi=120)
    ax.plot(x, values, color="#52616b", linewidth=0.45, alpha=0.9)
    if np.any(anomaly_mask):
        ax.scatter(
            x[anomaly_mask],
            values[anomaly_mask],
            s=7,
            color="#d7191c",
            alpha=0.75,
            label="candidate anomaly",
            zorder=3,
        )
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(
        f"{sid} GSR | n={len(values)} | anomalies={int(np.sum(anomaly_mask))} | "
        f"min={stats.get('min', math.nan):.1f}, median={stats.get('median', math.nan):.1f}, max={stats.get('max', math.nan):.1f}",
        fontsize=10,
    )
    ax.set_xlabel("Minutes from first processed GSR sample")
    ax.set_ylabel("Resistance (Kohm)")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)
    return png


def plot_overview(summary_rows, histogram_values):
    if summary_rows:
        top = sorted(summary_rows, key=lambda r: int(r["AnomalyRows"]), reverse=True)[:30]
        labels = [r["Subject"] for r in top]
        counts = [int(r["AnomalyRows"]) for r in top]
        fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
        ax.bar(labels, counts, color="#d7191c")
        ax.set_title("Top subjects by candidate anomaly rows")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=70)
        ax.grid(axis="y", linewidth=0.3, alpha=0.35)
        fig.tight_layout()
        fig.savefig(OVERVIEW / "top_anomaly_subjects.png")
        plt.close(fig)

    if len(histogram_values):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        ax.hist(histogram_values, bins=240, color="#52616b", alpha=0.85)
        ax.set_yscale("log")
        ax.set_title("GSR value distribution across processed combined files")
        ax.set_xlabel("Resistance (Kohm)")
        ax.set_ylabel("Count (log scale)")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        fig.tight_layout()
        fig.savefig(OVERVIEW / "global_value_distribution.png")
        plt.close(fig)


def build_index(summary_rows):
    rows = []
    for row in sorted(summary_rows, key=lambda r: r["Subject"]):
        sid = row["Subject"]
        plot_rel = f"per_subject_plots/{sid}_GSR.png"
        rows.append(
            "<tr>"
            f"<td>{escape(sid)}</td>"
            f"<td>{escape(row['Rows'])}</td>"
            f"<td>{escape(row['AnomalyRows'])}</td>"
            f"<td>{escape(row['AnomalyFraction'])}</td>"
            f"<td>{escape(row['Min'])}</td>"
            f"<td>{escape(row['Median'])}</td>"
            f"<td>{escape(row['Max'])}</td>"
            f"<td>{escape(row['Flags'])}</td>"
            f"<td><a href='{plot_rel}'><img src='{plot_rel}' width='320'></a></td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>GSR anomaly analysis</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; vertical-align: top; }}
th {{ background: #f2f2f2; position: sticky; top: 0; }}
img {{ display: block; }}
</style>
</head>
<body>
<h1>GSR anomaly analysis</h1>
<p>Rules: value &lt;= 0, value == 1, value &gt;= 1000, adjacent step &gt;= 500, or robust MAD outlier &gt; 8.</p>
<p><a href="overview/top_anomaly_subjects.png">Top anomaly subjects</a> | <a href="overview/global_value_distribution.png">Global value distribution</a> | <a href="reports/gsr_anomaly_summary.csv">Summary CSV</a> | <a href="reports/gsr_anomaly_points.csv">Anomaly points CSV</a></p>
<table>
<thead><tr><th>Subject</th><th>Rows</th><th>AnomalyRows</th><th>AnomalyFraction</th><th>Min</th><th>Median</th><th>Max</th><th>Flags</th><th>Plot</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def main():
    summary_rows = []
    anomaly_rows = []
    histogram_values = []

    for n in range(1, 159):
        sid = f"T{n:03d}"
        path = ROOT / sid / "processed_signal_data" / "GSR" / f"{sid}_GSR_T12_T22_T32_processed.csv"
        if not path.exists():
            continue

        timestamps, values, trusts, parse_errors = read_subject(path)
        reasons, stats = detect_anomalies(values)
        plot_subject(sid, timestamps, values, reasons, stats)

        if len(values):
            histogram_values.append(values[np.isfinite(values)])

        anomaly_mask = [bool(r) for r in reasons]
        reason_counts = stats.get("reason_counts", Counter())
        flags = []
        if stats.get("zero_or_one_fraction", 0) >= 0.05:
            flags.append("many_zero_or_one")
        if stats.get("high_ge_1000_fraction", 0) >= 0.05:
            flags.append("many_high_ge_1000")
        if stats.get("mode_fraction", 0) >= 0.8:
            flags.append(f"flatline_mode_{stats.get('mode_value', ''):.1f}")
        if len(values) == 0:
            flags.append("empty_target_rows")

        summary_rows.append(
            {
                "Subject": sid,
                "Rows": str(len(values)),
                "AnomalyRows": str(sum(anomaly_mask)),
                "AnomalyFraction": f"{(sum(anomaly_mask) / len(values)):.6f}" if len(values) else "",
                "Min": f"{stats.get('min', math.nan):.6g}" if len(values) else "",
                "P01": f"{stats.get('p01', math.nan):.6g}" if len(values) else "",
                "Median": f"{stats.get('median', math.nan):.6g}" if len(values) else "",
                "P99": f"{stats.get('p99', math.nan):.6g}" if len(values) else "",
                "Max": f"{stats.get('max', math.nan):.6g}" if len(values) else "",
                "ModeValue": f"{stats.get('mode_value', math.nan):.6g}" if len(values) else "",
                "ModeFraction": f"{stats.get('mode_fraction', 0):.6f}" if len(values) else "",
                "ZeroOrOneFraction": f"{stats.get('zero_or_one_fraction', 0):.6f}" if len(values) else "",
                "HighGe1000Fraction": f"{stats.get('high_ge_1000_fraction', 0):.6f}" if len(values) else "",
                "NonPositiveRows": str(reason_counts.get("non_positive_le_0", 0)),
                "SentinelOneRows": str(reason_counts.get("sentinel_value_1", 0)),
                "HighGe1000Rows": str(reason_counts.get("high_resistance_ge_1000", 0)),
                "LargeStepRows": str(reason_counts.get("large_step_ge_500", 0)),
                "RobustOutlierRows": str(reason_counts.get("robust_outlier_mad_gt_8", 0)),
                "ParseErrors": str(len(parse_errors)),
                "Flags": "|".join(flags),
                "SourceFile": str(path),
                "PlotFile": str(PLOTS / f"{sid}_GSR.png"),
            }
        )

        for idx, reason_set in enumerate(reasons):
            if not reason_set:
                continue
            anomaly_rows.append(
                {
                    "Subject": sid,
                    "RowNumber": str(idx + 2),
                    "Timestamp": timestamps[idx].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "Data": f"{values[idx]:.12g}",
                    "Trust": trusts[idx],
                    "Reasons": "|".join(sorted(reason_set)),
                    "SourceFile": str(path),
                }
            )

        for parse_error in parse_errors:
            anomaly_rows.append(
                {
                    "Subject": sid,
                    "RowNumber": str(parse_error["Row"]),
                    "Timestamp": parse_error["Timestamp"],
                    "Data": parse_error["Data"],
                    "Trust": parse_error["Trust"],
                    "Reasons": parse_error["Reason"],
                    "SourceFile": str(path),
                }
            )

    if histogram_values:
        histogram_values = np.concatenate(histogram_values)
    else:
        histogram_values = np.array([], dtype=float)

    write_csv(
        REPORTS / "gsr_anomaly_summary.csv",
        [
            "Subject",
            "Rows",
            "AnomalyRows",
            "AnomalyFraction",
            "Min",
            "P01",
            "Median",
            "P99",
            "Max",
            "ModeValue",
            "ModeFraction",
            "ZeroOrOneFraction",
            "HighGe1000Fraction",
            "NonPositiveRows",
            "SentinelOneRows",
            "HighGe1000Rows",
            "LargeStepRows",
            "RobustOutlierRows",
            "ParseErrors",
            "Flags",
            "SourceFile",
            "PlotFile",
        ],
        summary_rows,
    )
    write_csv(
        REPORTS / "gsr_anomaly_points.csv",
        ["Subject", "RowNumber", "Timestamp", "Data", "Trust", "Reasons", "SourceFile"],
        anomaly_rows,
    )
    plot_overview(summary_rows, histogram_values)
    build_index(summary_rows)

    print("subjects_with_gsr_files", len(summary_rows))
    print("subjects_with_nonempty_target_rows", sum(1 for r in summary_rows if int(r["Rows"]) > 0))
    print("total_rows", sum(int(r["Rows"]) for r in summary_rows))
    print("total_anomaly_rows", sum(int(r["AnomalyRows"]) for r in summary_rows))
    print("output_dir", OUT)
    for row in sorted(summary_rows, key=lambda r: int(r["AnomalyRows"]), reverse=True)[:20]:
        print(row["Subject"], row["Rows"], row["AnomalyRows"], row["AnomalyFraction"], row["Flags"], row["Min"], row["Median"], row["Max"])


if __name__ == "__main__":
    main()
