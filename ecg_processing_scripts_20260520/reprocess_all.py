"""
重新切割 T001-T158 的 ECG 和 GSR 数据。
根据原始数据 Trigger 列筛选 T12/T22/T32，并重采样到 200Hz (ECG) / 50Hz (GSR)。
ECG: 50通道取均值后插值到 200Hz
GSR: 直接插值到 50Hz
支持多个 CSV 文件拼接。
"""

import csv
import os
import glob
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = r"C:\Users\26354\Desktop\saili_data"

TARGET_LABELS = ["T12", "T22", "T32"]
TRUST_MAP = {"T12": "低", "T22": "中", "T32": "高"}
ECG_FS = 200  # Hz
GSR_FS = 50   # Hz

RAW_FMT = "%y-%m-%d %H:%M:%S.%f"
OUT_FMT = "%Y-%m-%d %H:%M:%S.%f"


def find_raw_files(subject_dir):
    """Return sorted ECG and GSR CSV paths for a subject."""
    ecg_dir = os.path.join(subject_dir, "ECG")
    gsr_dir = os.path.join(subject_dir, "GSR")
    ecg_csvs = sorted(glob.glob(os.path.join(ecg_dir, "*.csv")))
    gsr_csvs = sorted(glob.glob(os.path.join(gsr_dir, "*.csv")))
    if not ecg_csvs or not gsr_csvs:
        return None, None
    return ecg_csvs, gsr_csvs


def read_ecg_raw_multi(csv_paths):
    """Read and concatenate multiple ECG CSV files."""
    all_times = []
    all_signals = []
    for path in csv_paths:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for r in reader:
                t = datetime.strptime(r[0], RAW_FMT)
                ch_vals = [float(r[i]) for i in range(1, 51)]
                all_times.append(t)
                all_signals.append(np.mean(ch_vals))
    return all_times, np.array(all_signals, dtype=np.float64)


def read_gsr_raw_multi(csv_paths):
    """Read and concatenate multiple GSR CSV files."""
    all_times = []
    all_signals = []
    for path in csv_paths:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                t = datetime.strptime(r[0], RAW_FMT)
                all_times.append(t)
                all_signals.append(float(r[1]))
    return all_times, np.array(all_signals, dtype=np.float64)


def get_trigger_rows_multi(csv_paths):
    """Get trigger row indices across multiple concatenated CSV files."""
    triggers = {}
    offset = 0
    for path in csv_paths:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            for i, r in enumerate(reader):
                label = r[-1].strip()
                if label and label != "null" and label != "":
                    triggers.setdefault(label, []).append(offset + i)
        # Count rows in this file for offset
        with open(path, encoding="utf-8-sig") as f:
            row_count = sum(1 for _ in f) - 1  # minus header
        offset += row_count
    return triggers


def extract_segment(all_times, all_signal, trigger_indices, fs):
    """
    Extract segment by trigger indices and resample to target fs.
    Returns (datetime_strings, signal_array) or (None, None).
    """
    if not trigger_indices or len(trigger_indices) < 2:
        return None, None

    seg_times = [all_times[i] for i in trigger_indices]
    seg_signal = all_signal[trigger_indices]

    t0 = seg_times[0]
    t_seconds = np.array([(t - t0).total_seconds() for t in seg_times], dtype=np.float64)
    duration = t_seconds[-1]

    # Remove duplicate time points
    unique_mask = np.concatenate([[True], np.diff(t_seconds) > 0])
    if np.sum(unique_mask) < 2:
        return None, None

    t_unique = t_seconds[unique_mask]
    s_unique = seg_signal[unique_mask]

    n_samples = int(np.round(duration * fs)) + 1
    t_uniform = np.arange(n_samples, dtype=np.float64) / fs

    uniform_signal = np.interp(t_uniform, t_unique, s_unique)

    dt_strs = []
    for sec in t_uniform:
        dt = t0 + timedelta(seconds=float(sec))
        dt_strs.append(dt.strftime(OUT_FMT)[:-3])

    return dt_strs, uniform_signal


def process_subject(subject_dir):
    subject_id = os.path.basename(subject_dir)
    ecg_csvs, gsr_csvs = find_raw_files(subject_dir)

    if ecg_csvs is None or gsr_csvs is None:
        print(f"  [{subject_id}] SKIP: missing raw ECG or GSR files")
        return False

    n_ecg = len(ecg_csvs)
    n_gsr = len(gsr_csvs)
    file_info = f"({n_ecg} ECG + {n_gsr} GSR files)" if n_ecg > 1 or n_gsr > 1 else ""

    try:
        ecg_times, ecg_signal = read_ecg_raw_multi(ecg_csvs)
        gsr_times, gsr_signal = read_gsr_raw_multi(gsr_csvs)
    except Exception as e:
        print(f"  [{subject_id}] ERROR reading raw data: {e}")
        return False

    ecg_triggers = get_trigger_rows_multi(ecg_csvs)
    gsr_triggers = get_trigger_rows_multi(gsr_csvs)

    print(f"  [{subject_id}] {file_info} ECG triggers: {sorted(ecg_triggers.keys())}, GSR triggers: {sorted(gsr_triggers.keys())}")

    out_ecg_dir = os.path.join(subject_dir, "processed_signal_data", "ECG")
    out_gsr_dir = os.path.join(subject_dir, "processed_signal_data", "GSR")
    os.makedirs(out_ecg_dir, exist_ok=True)
    os.makedirs(out_gsr_dir, exist_ok=True)

    any_success = False

    for label in TARGET_LABELS:
        trust = TRUST_MAP[label]
        tag = {"T12": "low", "T22": "medium", "T32": "high"}[label]

        ecg_ok = False
        if label in ecg_triggers and len(ecg_triggers[label]) > 1:
            dt_strs, signal = extract_segment(ecg_times, ecg_signal, ecg_triggers[label], ECG_FS)
            if dt_strs is not None:
                out_path = os.path.join(out_ecg_dir, f"{subject_id}_ECG_{label}_{tag}_trust.csv")
                with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Data", "Trust"])
                    for ts, val in zip(dt_strs, signal):
                        writer.writerow([ts, f"{val:.6f}", trust])
                ecg_ok = True
                print(f"    ECG {label}: {len(dt_strs)} samples @ {ECG_FS}Hz -> {os.path.basename(out_path)}")
        else:
            print(f"    ECG {label}: SKIP (trigger rows: {len(ecg_triggers.get(label, []))})")

        gsr_ok = False
        if label in gsr_triggers and len(gsr_triggers[label]) > 1:
            dt_strs, signal = extract_segment(gsr_times, gsr_signal, gsr_triggers[label], GSR_FS)
            if dt_strs is not None:
                out_path = os.path.join(out_gsr_dir, f"{subject_id}_GSR_{label}_{tag}_trust.csv")
                with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Data", "Trust"])
                    for ts, val in zip(dt_strs, signal):
                        writer.writerow([ts, f"{val:.6f}", trust])
                gsr_ok = True
                print(f"    GSR {label}: {len(dt_strs)} samples @ {GSR_FS}Hz -> {os.path.basename(out_path)}")
        else:
            print(f"    GSR {label}: SKIP (trigger rows: {len(gsr_triggers.get(label, []))})")

        if ecg_ok and gsr_ok:
            any_success = True

    return any_success


def main():
    all_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "T[0-9]*")))
    print(f"Found {len(all_dirs)} subject directories\n")

    success_count = 0
    skip_count = 0

    for subject_dir in all_dirs:
        subj = os.path.basename(subject_dir)
        if not os.path.isdir(os.path.join(subject_dir, "ECG")) or \
           not os.path.isdir(os.path.join(subject_dir, "GSR")):
            print(f"  [{subj}] SKIP: no ECG/GSR subdirectories")
            skip_count += 1
            continue

        ok = process_subject(subject_dir)
        if ok:
            success_count += 1
        else:
            skip_count += 1
        print()

    print(f"Done. Success: {success_count}, Skipped/Errors: {skip_count}")


if __name__ == "__main__":
    main()
