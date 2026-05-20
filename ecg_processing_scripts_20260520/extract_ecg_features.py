"""
Extract ECG signal features from processed ECG CSV files.
Equivalent to extract_scr_features.py but for the ECG modality.

The raw "ECG" data in this project is a 50-channel sensor array sampled at ~4 Hz.
The processed data is channel-averaged and interpolated to 200 Hz.
This script detects physiological events (periodic peaks), extracts interval
statistics, performs frequency analysis, and generates per-condition plots.

Reads per-subject per-condition processed ECG CSV files from
  Txxx/processed_signal_data/ECG/*.csv
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

BASE_DIR = Path(r"C:\Users\26354\Desktop\saili_data")
OUTPUT_DIR = BASE_DIR / "ICSlab_project_ECG_output"
PLOTS_DIR = OUTPUT_DIR / "ECG_plots"
FEATURES_CSV = OUTPUT_DIR / "ECG_HRV_features.csv"

ECG_FS = 200.0  # Hz (interpolated)

CONDITION_LABELS = {
    "T12": "low_trust",
    "T22": "medium_trust",
    "T32": "high_trust",
}


def read_ecg_csv(path: Path) -> tuple[np.ndarray, str, float]:
    """Read processed ECG CSV. Returns (data_array, trust_label, duration_sec)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    data = df["Data"].to_numpy(dtype=float)
    trust = str(df["Trust"].iloc[0]) if "Trust" in df.columns else "unknown"
    duration = len(data) / ECG_FS
    return data, trust, duration


def detect_peaks(data: np.ndarray, fs: float) -> dict:
    """
    Detect peaks in the signal using scipy.signal.find_peaks.
    Adapts to signal characteristics automatically.
    """
    n = len(data)
    result = {
        "n_peaks": 0, "peaks": np.array([], dtype=int),
        "mean_ibi_ms": np.nan, "std_ibi_ms": np.nan,
        "cv_ibi_pct": np.nan, "peak_rate_per_min": np.nan,
        "mean_peak_amplitude": np.nan, "std_peak_amplitude": np.nan,
    }

    if n < fs * 5:
        return result

    # Remove NaN/Inf and detrend
    clean = data.copy()
    bad = ~np.isfinite(clean)
    if bad.any():
        good_idx = np.flatnonzero(~bad)
        if len(good_idx) == 0:
            return result
        bad_idx = np.flatnonzero(bad)
        clean[bad_idx] = np.interp(bad_idx, good_idx, clean[good_idx])

    # Detrend to remove baseline wander
    clean = scipy_signal.detrend(clean)

    # Bandpass filter to isolate pulsatile component (0.5 - 5 Hz)
    nyq = fs / 2.0
    try:
        b, a = scipy_signal.butter(3, [0.5 / nyq, 5.0 / nyq], btype="band")
        filtered = scipy_signal.filtfilt(b, a, clean)
    except Exception:
        filtered = clean

    # Adaptive peak detection
    signal_std = float(np.std(filtered))
    if signal_std < 1e-9:
        return result

    # Minimum distance between peaks: at least 0.33s (max ~180 BPM)
    min_distance = max(1, int(fs * 0.33))
    # Prominence: adaptive based on signal std
    prominence = max(signal_std * 0.3, np.percentile(np.abs(filtered), 90) * 0.5)

    peaks, props = scipy_signal.find_peaks(
        filtered,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peaks) < 2:
        # Try with lower threshold
        peaks, props = scipy_signal.find_peaks(
            filtered,
            distance=max(1, int(fs * 0.5)),
            prominence=signal_std * 0.15,
        )

    if len(peaks) < 2:
        result["n_peaks"] = len(peaks)
        result["peaks"] = peaks
        return result

    # Compute inter-beat intervals
    ibi = np.diff(peaks) / fs * 1000.0  # ms
    peak_rate = 60000.0 / np.mean(ibi) if np.mean(ibi) > 0 else np.nan

    result["n_peaks"] = len(peaks)
    result["peaks"] = peaks
    result["mean_ibi_ms"] = float(np.mean(ibi))
    result["std_ibi_ms"] = float(np.std(ibi, ddof=1)) if len(ibi) > 1 else 0.0
    result["cv_ibi_pct"] = float(np.std(ibi, ddof=1) / np.mean(ibi) * 100) if len(ibi) > 1 and np.mean(ibi) > 0 else np.nan
    result["peak_rate_per_min"] = float(peak_rate)
    result["mean_peak_amplitude"] = float(np.mean(props["prominences"])) if "prominences" in props else np.nan
    result["std_peak_amplitude"] = float(np.std(props["prominences"])) if "prominences" in props else np.nan

    return result


def compute_signal_frequency_features(data: np.ndarray, fs: float) -> dict:
    """Compute frequency-domain features of the signal directly using Welch's method."""
    clean = data.copy()
    bad = ~np.isfinite(clean)
    if bad.any():
        good_idx = np.flatnonzero(~bad)
        if len(good_idx) < 2:
            return {"lf_power": np.nan, "hf_power": np.nan, "lf_hf_ratio": np.nan,
                    "total_power": np.nan, "dominant_freq_hz": np.nan,
                    "spectral_entropy": np.nan}
        bad_idx = np.flatnonzero(bad)
        clean[bad_idx] = np.interp(bad_idx, good_idx, clean[good_idx])

    clean = scipy_signal.detrend(clean)

    try:
        nperseg = min(int(fs * 4), len(clean))
        nperseg = max(64, 2 ** int(np.ceil(np.log2(nperseg))))
        freqs, psd = scipy_signal.welch(
            clean, fs=fs, nperseg=nperseg,
            nfft=max(256, 2 ** int(np.ceil(np.log2(len(clean))))),
        )
    except Exception:
        return {"lf_power": np.nan, "hf_power": np.nan, "lf_hf_ratio": np.nan,
                "total_power": np.nan, "dominant_freq_hz": np.nan,
                "spectral_entropy": np.nan}

    # Physiological frequency bands
    lf_band = (freqs >= 0.5) & (freqs < 2.0)
    hf_band = (freqs >= 2.0) & (freqs < 5.0)
    total_band = (freqs >= 0.1) & (freqs < 10.0)

    lf = float(np.trapezoid(psd[lf_band], freqs[lf_band])) if np.any(lf_band) else np.nan
    hf = float(np.trapezoid(psd[hf_band], freqs[hf_band])) if np.any(hf_band) else np.nan
    total = float(np.trapezoid(psd[total_band], freqs[total_band])) if np.any(total_band) else np.nan

    # Dominant frequency (in physiological range 0.5-5 Hz)
    phys_band = (freqs >= 0.5) & (freqs < 5.0)
    if np.any(phys_band):
        dominant_freq = float(freqs[phys_band][np.argmax(psd[phys_band])])
    else:
        dominant_freq = np.nan

    # Spectral entropy
    if np.any(phys_band) and total and total > 0:
        psd_norm = psd[phys_band] / np.sum(psd[phys_band])
        psd_norm = psd_norm[psd_norm > 0]
        entropy = -np.sum(psd_norm * np.log(psd_norm))
        entropy_norm = entropy / np.log(len(psd_norm)) if len(psd_norm) > 1 else np.nan
    else:
        entropy_norm = np.nan

    return {
        "lf_power": lf,
        "hf_power": hf,
        "lf_hf_ratio": float(lf / hf) if lf and hf and hf > 0 else np.nan,
        "total_power": total,
        "dominant_freq_hz": dominant_freq,
        "spectral_entropy": entropy_norm,
    }


def plot_ecg(subject_id: str, task_id: str, data: np.ndarray,
             peak_info: dict, freq_info: dict, duration_sec: float) -> Path:
    """Generate a per-condition ECG plot with peaks and frequency info."""
    subj_plot_dir = PLOTS_DIR / subject_id
    subj_plot_dir.mkdir(parents=True, exist_ok=True)
    task_lower = task_id.lower()
    png = subj_plot_dir / f"{subject_id}_{task_lower}_plot.png"

    n = len(data)
    if n < 2:
        fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
        ax.text(0.5, 0.5, f"No ECG data for {subject_id} {task_id}",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        return png

    t_sec = np.arange(n) / ECG_FS

    # Detrend and filter for display
    clean = scipy_signal.detrend(data)
    try:
        b, a = scipy_signal.butter(3, [0.5 / (ECG_FS / 2), 5.0 / (ECG_FS / 2)], btype="band")
        filtered = scipy_signal.filtfilt(b, a, clean)
    except Exception:
        filtered = clean

    peaks = peak_info.get("peaks", np.array([], dtype=int))

    fig = plt.figure(figsize=(14, 8), dpi=120)
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1])

    # Top-left: Raw signal with filtered overlay and peaks
    ax_sig = fig.add_subplot(gs[0, :])
    ax_sig.plot(t_sec, data, color="#b0bec5", linewidth=0.3, alpha=0.6, label="Raw")
    ax_sig.plot(t_sec, filtered, color="#263238", linewidth=0.6, alpha=0.9, label="Filtered (0.5-5 Hz)")

    if len(peaks) > 0:
        valid = peaks[peaks < n]
        ax_sig.scatter(t_sec[valid], filtered[valid],
                       s=18, color="#d7191c", alpha=0.85, zorder=3,
                       label=f"Peaks ({len(valid)})")

    ax_sig.set_ylabel("Amplitude")
    condition = CONDITION_LABELS.get(task_id, "")
    ax_sig.set_title(
        f"{subject_id}  {task_id} ({condition}) | "
        f"dur={duration_sec:.1f}s | "
        f"Peak rate={peak_info.get('peak_rate_per_min', np.nan):.1f}/min | "
        f"Mean IBI={peak_info.get('mean_ibi_ms', np.nan):.0f}ms",
        fontsize=10,
    )
    ax_sig.legend(loc="upper right", fontsize=7, frameon=False, ncol=3)
    ax_sig.grid(True, linewidth=0.3, alpha=0.35)

    # Middle-left: Zoom to first 60s
    ax_zoom = fig.add_subplot(gs[1, 0])
    zoom_end = min(int(ECG_FS * 60), n)
    ax_zoom.plot(t_sec[:zoom_end], filtered[:zoom_end], color="#263238", linewidth=0.6)
    if len(peaks) > 0:
        zoom_peaks = peaks[(peaks < zoom_end)]
        ax_zoom.scatter(t_sec[zoom_peaks], filtered[zoom_peaks],
                        s=12, color="#d7191c", alpha=0.8, zorder=3)
    ax_zoom.set_xlabel("Time (s)")
    ax_zoom.set_ylabel("Filtered")
    ax_zoom.set_title(f"First {min(60, int(duration_sec))}s zoom", fontsize=9)
    ax_zoom.grid(True, linewidth=0.3, alpha=0.35)

    # Middle-right: Inter-beat interval distribution
    ax_ibi = fig.add_subplot(gs[1, 1])
    if len(peaks) >= 2:
        ibi = np.diff(peaks[peaks < n]) / ECG_FS * 1000.0
        ax_ibi.hist(ibi, bins=min(30, len(ibi) // 2), color="#4caf50", alpha=0.7, edgecolor="#263238", linewidth=0.5)
        ax_ibi.axvline(x=np.mean(ibi), color="#d7191c", linestyle="--", linewidth=1.0,
                       label=f"Mean: {np.mean(ibi):.0f} ms")
        ax_ibi.legend(fontsize=7, frameon=False)
    ax_ibi.set_xlabel("IBI (ms)")
    ax_ibi.set_ylabel("Count")
    ax_ibi.set_title("Inter-Beat Interval Distribution", fontsize=9)
    ax_ibi.grid(True, linewidth=0.3, alpha=0.35)

    # Bottom-left: Frequency spectrum
    ax_psd = fig.add_subplot(gs[2, 0])
    clean_sig = scipy_signal.detrend(data)
    bad = ~np.isfinite(clean_sig)
    if bad.any():
        good_idx = np.flatnonzero(~bad)
        bad_idx = np.flatnonzero(bad)
        clean_sig[bad_idx] = np.interp(bad_idx, good_idx, clean_sig[good_idx])
    try:
        nperseg = min(int(ECG_FS * 4), len(clean_sig))
        nperseg = max(64, 2 ** int(np.ceil(np.log2(nperseg))))
        freqs, psd = scipy_signal.welch(clean_sig, fs=ECG_FS, nperseg=nperseg)
        mask = (freqs >= 0.1) & (freqs <= 10)
        ax_psd.semilogy(freqs[mask], psd[mask], color="#263238", linewidth=0.8)
        ax_psd.axvspan(0.5, 2.0, alpha=0.1, color="#2196f3", label="LF (0.5-2 Hz)")
        ax_psd.axvspan(2.0, 5.0, alpha=0.1, color="#4caf50", label="HF (2-5 Hz)")
        ax_psd.legend(fontsize=7, frameon=False, loc="upper right")
    except Exception:
        ax_psd.text(0.5, 0.5, "PSD computation failed", ha="center", va="center")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power")
    ax_psd.set_title(
        f"Dominant freq: {freq_info.get('dominant_freq_hz', np.nan):.2f} Hz | "
        f"LF/HF: {freq_info.get('lf_hf_ratio', np.nan):.2f}",
        fontsize=9,
    )
    ax_psd.grid(True, linewidth=0.3, alpha=0.35)

    # Bottom-right: Signal stats text panel
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.set_axis_off()
    stats_lines = [
        f"Samples: {n}",
        f"Duration: {duration_sec:.1f} s",
        f"Mean: {np.nanmean(data):.4f}",
        f"Std: {np.nanstd(data):.4f}",
        f"Min: {np.nanmin(data):.4f}",
        f"Max: {np.nanmax(data):.4f}",
        f"---",
        f"Peaks: {peak_info.get('n_peaks', 0)}",
        f"Peak rate: {peak_info.get('peak_rate_per_min', np.nan):.1f}/min",
        f"Mean IBI: {peak_info.get('mean_ibi_ms', np.nan):.0f} ms",
        f"IBI CV: {peak_info.get('cv_ibi_pct', np.nan):.1f}%",
        f"---",
        f"LF/HF ratio: {freq_info.get('lf_hf_ratio', np.nan):.3f}",
    ]
    for i, line in enumerate(stats_lines):
        ax_stats.text(0.05, 0.95 - i * 0.07, line, transform=ax_stats.transAxes,
                      fontsize=8, fontfamily="monospace", va="top")

    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return png


def summarize_file(path: Path) -> dict[str, object] | None:
    """Process one ECG CSV file and extract signal features."""
    try:
        data, trust, duration_sec = read_ecg_csv(path)
    except Exception as exc:
        print(f"  [ERROR] Failed to read {path}: {exc}")
        return None

    task_id = path.stem.split("_")[2] if "_" in path.stem else ""
    # path = Txxx/processed_signal_data/ECG/file.csv -> user_id = Txxx
    user_id = str(path.parent.parent.parent.name)

    MIN_DURATION = 10.0  # minimum 10 seconds

    if len(data) < ECG_FS * MIN_DURATION:
        print(f"  [WARN] {path.name}: too short ({len(data)} samples, {duration_sec:.1f}s)")
        return {
            "user_id": user_id,
            "task_id": task_id,
            "condition": CONDITION_LABELS.get(task_id, "unknown"),
            "file": str(path),
            "n_samples": int(len(data)),
            "sampling_rate_hz": ECG_FS,
            "duration_sec": duration_sec,
            "ecg_mean": float(np.nanmean(data)) if len(data) > 0 else np.nan,
            "ecg_std": float(np.nanstd(data)) if len(data) > 0 else np.nan,
            "ecg_min": float(np.nanmin(data)) if len(data) > 0 else np.nan,
            "ecg_max": float(np.nanmax(data)) if len(data) > 0 else np.nan,
            "n_peaks": 0,
            "quality_flags": "too_short",
        }

    # Peak detection
    peak_info = detect_peaks(data, ECG_FS)

    # Frequency analysis
    freq_info = compute_signal_frequency_features(data, ECG_FS)

    # Quality flags
    flags = []
    if duration_sec < 60:
        flags.append("short_duration")
    data_std = float(np.nanstd(data))
    if data_std < 0.0005:  # nearly flat DC signal
        flags.append("flatline_signal")
    if data_std > 1.0:  # unusually large variation
        flags.append("high_variation")
    if np.nanmax(np.abs(data)) > 2.0:  # beyond typical range for this sensor
        flags.append("extreme_amplitude")
    if peak_info["n_peaks"] < 3:
        flags.append("too_few_peaks")
    rate = peak_info.get("peak_rate_per_min", np.nan)
    if not np.isnan(rate) and (rate < 20 or rate > 200):
        flags.append("abnormal_peak_rate")

    features = {
        "user_id": user_id,
        "task_id": task_id,
        "condition": CONDITION_LABELS.get(task_id, "unknown"),
        "file": str(path),
        "n_samples": int(len(data)),
        "sampling_rate_hz": ECG_FS,
        "duration_sec": duration_sec,
        "ecg_mean": float(np.nanmean(data)),
        "ecg_std": float(np.nanstd(data)),
        "ecg_min": float(np.nanmin(data)),
        "ecg_max": float(np.nanmax(data)),
        "n_peaks": peak_info["n_peaks"],
        "peak_rate_per_min": peak_info["peak_rate_per_min"],
        "mean_ibi_ms": peak_info["mean_ibi_ms"],
        "std_ibi_ms": peak_info["std_ibi_ms"],
        "cv_ibi_pct": peak_info["cv_ibi_pct"],
        "mean_peak_amplitude": peak_info["mean_peak_amplitude"],
        "std_peak_amplitude": peak_info["std_peak_amplitude"],
        "lf_power": freq_info["lf_power"],
        "hf_power": freq_info["hf_power"],
        "lf_hf_ratio": freq_info["lf_hf_ratio"],
        "total_power": freq_info["total_power"],
        "dominant_freq_hz": freq_info["dominant_freq_hz"],
        "spectral_entropy": freq_info["spectral_entropy"],
        "quality_flags": ";".join(flags),
    }

    # Generate plot
    try:
        png_path = plot_ecg(user_id, task_id, data, peak_info, freq_info, duration_sec)
        features["plot_file"] = str(png_path)
    except Exception as exc:
        print(f"  [WARN] Plot generation failed for {path.name}: {exc}")
        features["plot_file"] = ""

    rate_str = f"{peak_info['peak_rate_per_min']:.1f}/min" if not np.isnan(peak_info.get("peak_rate_per_min", np.nan)) else "N/A"
    print(f"  [{features['user_id']}] {features['task_id']}: "
          f"rate={rate_str}, "
          f"{peak_info['n_peaks']} peaks, "
          f"dur={duration_sec:.1f}s, "
          f"flags={';'.join(flags) if flags else 'none'}")

    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ECG signal features from processed CSV files.")
    parser.add_argument(
        "--input-dir", default=str(BASE_DIR),
        help="Base directory containing Txxx subject folders.",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Output directory for features CSV and plots.",
    )
    args = parser.parse_args()

    base = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    plots_root = out_dir / "ECG_plots"
    features_path = out_dir / "ECG_HRV_features.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(base.glob("T*/processed_signal_data/ECG/*.csv"))
    print(f"Found {len(csv_files)} ECG CSV files\n")

    rows = []
    for csv_path in csv_files:
        result = summarize_file(csv_path)
        if result is not None:
            rows.append(result)

    df = pd.DataFrame(rows)
    if "user_id" in df.columns:
        df["user_id_num"] = pd.to_numeric(df["user_id"].str.lstrip("T"), errors="coerce")
        df = df.sort_values(["user_id_num", "task_id"]).drop(columns=["user_id_num"])

    df.to_csv(features_path, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(df)} rows to {features_path}")

    if "quality_flags" in df.columns:
        print("\nQuality flag distribution:")
        print(df["quality_flags"].value_counts(dropna=False).to_string())

    print(f"\n--- Summary ---")
    print(f"Total files: {len(df)}")
    valid = df[df["quality_flags"].isna() | (df["quality_flags"] == "")]
    print(f"Valid (no flags): {len(valid)}")
    print(f"With flags: {len(df) - len(valid)}")

    if "peak_rate_per_min" in df.columns:
        rates = df["peak_rate_per_min"].dropna()
        if len(rates) > 0:
            print(f"Peak rate range: {rates.min():.1f} - {rates.max():.1f} /min")
            print(f"Peak rate mean: {rates.mean():.1f} /min")

    if "n_peaks" in df.columns:
        print(f"Files with <3 peaks: {(df['n_peaks'] < 3).sum()}")
        print(f"Files with 0 peaks: {(df['n_peaks'] == 0).sum()}")


if __name__ == "__main__":
    main()
