"""Process ECG and GSR CSV files with fixed sampling rates.

This script reads participant folders named T001..T158. It does not modify
raw ECG/GSR CSV files. It writes processed CSV files under each participant's
output folder.

Output columns are always:
    Timestamp,Data,Trust

Default trust mapping:
    T12 -> 低
    T22 -> 中
    T32 -> 高

Use --trust-label code to keep T12/T22/T32.
Use --trust-label en to write low/medium/high.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


TARGET_TRUSTS = ("T12", "T22", "T32")
TRUST_ZH = {"T12": "低", "T22": "中", "T32": "高"}
TRUST_EN = {"T12": "low", "T22": "medium", "T32": "high"}

ECG_SAMPLE_RATE_HZ = 200
GSR_SAMPLE_RATE_HZ = 50
ECG_SAMPLES_PER_ROW = 50


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    for fmt in (
        "%y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported timestamp: {value!r}")


def round_to_millisecond(dt: datetime) -> datetime:
    if dt.microsecond >= 999_500:
        return (dt + timedelta(seconds=1)).replace(microsecond=0)
    return dt.replace(microsecond=((dt.microsecond + 500) // 1000) * 1000)


def format_millisecond(dt: datetime) -> str:
    dt = round_to_millisecond(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def trust_value(code: str, mode: str) -> str:
    if mode == "code":
        return code
    if mode == "zh":
        return TRUST_ZH[code]
    if mode == "en":
        return TRUST_EN[code]
    raise ValueError(f"Unsupported trust label mode: {mode}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Data", "Trust"])
        writer.writeheader()
        writer.writerows(rows)


def group_by_trust(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["Trust"]].append(row)
    return grouped


def process_ecg_file(path: Path, trust_label: str) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        return []

    anchor = parse_timestamp(rows[0]["Timestamp"])
    sample_step = timedelta(seconds=1 / ECG_SAMPLE_RATE_HZ)
    ecg_columns = [f"raw ecg{i}" for i in range(1, ECG_SAMPLES_PER_ROW + 1)]
    output: list[dict[str, str]] = []
    sample_base_index = 0

    for row in rows:
        trigger = row.get("Trigger", "").strip()
        if trigger in TARGET_TRUSTS:
            for sample_offset, column in enumerate(ecg_columns):
                timestamp = anchor + (sample_base_index + sample_offset) * sample_step
                output.append(
                    {
                        "Timestamp": format_millisecond(timestamp),
                        "Data": row[column],
                        "Trust": trust_value(trigger, trust_label),
                    }
                )
        sample_base_index += ECG_SAMPLES_PER_ROW

    return output


def process_gsr_file(path: Path, trust_label: str) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        return []

    anchor = parse_timestamp(rows[0]["Timestamp"])
    sample_step = timedelta(seconds=1 / GSR_SAMPLE_RATE_HZ)
    output: list[dict[str, str]] = []

    for sample_index, row in enumerate(rows):
        trigger = row.get("Trigger", "").strip()
        if trigger in TARGET_TRUSTS:
            output.append(
                {
                    "Timestamp": format_millisecond(anchor + sample_index * sample_step),
                    "Data": row["Resistance(Koum)"],
                    "Trust": trust_value(trigger, trust_label),
                }
            )

    return output


def output_trust_name(trust_code: str) -> str:
    return {"T12": "low", "T22": "medium", "T32": "high"}[trust_code]


def process_subject(
    root: Path,
    subject_id: str,
    output_dir_name: str,
    trust_label: str,
    write_split_files: bool,
) -> dict[str, object]:
    subject_dir = root / subject_id
    ecg_files = sorted((subject_dir / "ECG").glob("*.csv")) if (subject_dir / "ECG").exists() else []
    gsr_files = sorted((subject_dir / "GSR").glob("*.csv")) if (subject_dir / "GSR").exists() else []

    if not ecg_files and not gsr_files:
        return {
            "Subject": subject_id,
            "Status": "no_input_csv",
            "ECGFiles": 0,
            "GSRFiles": 0,
            "ECGRows": 0,
            "GSRRows": 0,
        }

    out_base = subject_dir / output_dir_name
    ecg_rows: list[dict[str, str]] = []
    gsr_rows: list[dict[str, str]] = []

    for path in ecg_files:
        ecg_rows.extend(process_ecg_file(path, trust_label))
    if ecg_files:
        ecg_dir = out_base / "ECG"
        write_csv(ecg_dir / f"{subject_id}_ECG_T12_T22_T32_processed.csv", ecg_rows)
        if write_split_files:
            grouped = group_by_trust(ecg_rows)
            for trust_code in TARGET_TRUSTS:
                label = trust_value(trust_code, trust_label)
                name = output_trust_name(trust_code)
                write_csv(ecg_dir / f"{subject_id}_ECG_{trust_code}_{name}_trust.csv", grouped.get(label, []))

    for path in gsr_files:
        gsr_rows.extend(process_gsr_file(path, trust_label))
    if gsr_files:
        gsr_dir = out_base / "GSR"
        write_csv(gsr_dir / f"{subject_id}_GSR_T12_T22_T32_processed.csv", gsr_rows)
        if write_split_files:
            grouped = group_by_trust(gsr_rows)
            for trust_code in TARGET_TRUSTS:
                label = trust_value(trust_code, trust_label)
                name = output_trust_name(trust_code)
                write_csv(gsr_dir / f"{subject_id}_GSR_{trust_code}_{name}_trust.csv", grouped.get(label, []))

    return {
        "Subject": subject_id,
        "Status": "ok",
        "ECGFiles": len(ecg_files),
        "GSRFiles": len(gsr_files),
        "ECGRows": len(ecg_rows),
        "GSRRows": len(gsr_rows),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Subject", "Status", "ECGFiles", "GSRFiles", "ECGRows", "GSRRows"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process ECG/GSR files with fixed sampling rates.")
    parser.add_argument("--root", required=True, type=Path, help="Dataset root containing T001, T002, ... folders.")
    parser.add_argument("--start", default=1, type=int, help="First participant number.")
    parser.add_argument("--end", default=158, type=int, help="Last participant number, inclusive.")
    parser.add_argument("--output-dir-name", default="processed_signal_data", help="Output folder inside each participant.")
    parser.add_argument("--trust-label", choices=("code", "zh", "en"), default="zh")
    parser.add_argument("--no-split-files", action="store_true", help="Only write combined modality files.")
    parser.add_argument("--summary-csv", type=Path, help="Optional batch summary CSV path.")
    args = parser.parse_args()

    results = []
    for number in range(args.start, args.end + 1):
        subject_id = f"T{number:03d}"
        results.append(
            process_subject(
                root=args.root,
                subject_id=subject_id,
                output_dir_name=args.output_dir_name,
                trust_label=args.trust_label,
                write_split_files=not args.no_split_files,
            )
        )

    if args.summary_csv:
        write_summary(args.summary_csv, results)

    status_counts = Counter(str(row["Status"]) for row in results)
    print("subjects_total", len(results))
    print("status_counts", dict(status_counts))
    print("ecg_subjects", sum(1 for row in results if int(row["ECGFiles"]) > 0))
    print("gsr_subjects", sum(1 for row in results if int(row["GSRFiles"]) > 0))
    print("total_ecg_rows", sum(int(row["ECGRows"]) for row in results))
    print("total_gsr_rows", sum(int(row["GSRRows"]) for row in results))
    print("no_input", ",".join(str(row["Subject"]) for row in results if row["Status"] == "no_input_csv"))
    print(
        "missing_modality",
        [
            (row["Subject"], row["ECGFiles"], row["GSRFiles"])
            for row in results
            if row["Status"] == "ok" and (row["ECGFiles"] == 0 or row["GSRFiles"] == 0)
        ],
    )


if __name__ == "__main__":
    main()
