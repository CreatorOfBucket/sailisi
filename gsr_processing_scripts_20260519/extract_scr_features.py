from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal


CONDITION_LABELS = {
    "t12": "low_trust",
    "t22": "medium_trust",
    "t32": "high_trust",
}


@dataclass
class ScrEvent:
    onset_idx: int
    peak_idx: int
    recovery_idx: int | None
    amplitude: float
    rise_time_sec: float
    half_recovery_time_sec: float | None


def _odd_window(samples: int, max_len: int) -> int:
    samples = max(3, int(samples))
    if samples % 2 == 0:
        samples += 1
    if samples > max_len:
        samples = max_len if max_len % 2 == 1 else max_len - 1
    return max(samples, 3)


def parse_timestamps(values: np.ndarray) -> list[datetime]:
    parsed = []
    for value in values:
        text = str(value)
        for fmt in ("%y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                parsed.append(datetime.strptime(text, fmt))
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unsupported timestamp format: {text}")
    return parsed


def estimate_sampling_rate(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 50.0
    parsed = parse_timestamps(timestamps)
    diffs = np.array(
        [(b - a).total_seconds() for a, b in zip(parsed[:-1], parsed[1:])],
        dtype=float,
    )
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 50.0
    return float(1.0 / np.median(diffs))


def interpolate_bad_values(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=float).copy()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x
    good_idx = np.flatnonzero(~bad)
    if good_idx.size == 0:
        return np.zeros_like(x)
    bad_idx = np.flatnonzero(bad)
    x[bad_idx] = np.interp(bad_idx, good_idx, x[good_idx])
    return x


def decompose_phasic(data: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = interpolate_bad_values(data)
    if x.size < 5:
        return x.copy(), np.zeros_like(x)

    tonic_window = _odd_window(round(fs * 20), x.size)
    polyorder = 3 if tonic_window >= 7 else 1
    tonic = signal.savgol_filter(x, tonic_window, polyorder, mode="interp")
    phasic = x - tonic

    smooth_window = _odd_window(round(fs * 0.5), phasic.size)
    if smooth_window >= 5:
        phasic = signal.savgol_filter(phasic, smooth_window, 2, mode="interp")
    return tonic, phasic


def detect_scr_events(
    phasic: np.ndarray,
    fs: float,
    min_amplitude: float = 0.01,
    min_distance_sec: float = 1.0,
) -> list[ScrEvent]:
    if phasic.size < max(5, int(fs * 2)):
        return []

    spread = float(np.nanstd(phasic))
    prominence = max(min_amplitude, 0.2 * spread)
    peaks, _ = signal.find_peaks(
        phasic,
        prominence=prominence,
        distance=max(1, int(round(fs * min_distance_sec))),
    )

    events: list[ScrEvent] = []
    onset_back = max(1, int(round(fs * 4)))
    recovery_forward = max(1, int(round(fs * 10)))

    for peak_idx in peaks:
        left = max(0, peak_idx - onset_back)
        onset_idx = left + int(np.argmin(phasic[left : peak_idx + 1]))
        amplitude = float(phasic[peak_idx] - phasic[onset_idx])
        if not np.isfinite(amplitude) or amplitude < min_amplitude:
            continue

        half_level = phasic[onset_idx] + amplitude / 2.0
        right = min(phasic.size, peak_idx + recovery_forward + 1)
        recovery_candidates = np.flatnonzero(phasic[peak_idx:right] <= half_level)
        recovery_idx = None
        half_recovery = None
        if recovery_candidates.size:
            recovery_idx = peak_idx + int(recovery_candidates[0])
            half_recovery = float((recovery_idx - peak_idx) / fs)

        events.append(
            ScrEvent(
                onset_idx=onset_idx,
                peak_idx=int(peak_idx),
                recovery_idx=recovery_idx,
                amplitude=amplitude,
                rise_time_sec=float((peak_idx - onset_idx) / fs),
                half_recovery_time_sec=half_recovery,
            )
        )
    return events


def summarize_file(path: Path, min_amplitude: float) -> dict[str, object]:
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    mat_dict = mat["mat_dict"]
    data = np.asarray(mat_dict.data, dtype=float).ravel()
    timestamps = np.asarray(mat_dict.Timestamp).ravel()
    task_id = str(mat_dict.task_id)
    user_id = str(mat_dict.user_id)
    fs = estimate_sampling_rate(timestamps)
    duration_sec = float(data.size / fs) if fs > 0 else np.nan

    tonic, phasic = decompose_phasic(data, fs)
    events = detect_scr_events(phasic, fs, min_amplitude=min_amplitude)

    amplitudes = np.array([event.amplitude for event in events], dtype=float)
    rise_times = np.array([event.rise_time_sec for event in events], dtype=float)
    recoveries = np.array(
        [
            event.half_recovery_time_sec
            for event in events
            if event.half_recovery_time_sec is not None
        ],
        dtype=float,
    )
    positive_phasic = np.clip(phasic, 0, None)

    flags = []
    if duration_sec < 60:
        flags.append("short_duration")
    if np.nanmax(np.abs(data)) > 10000 or np.nanstd(data) > 1000:
        flags.append("extreme_signal")
    if len(events) == 0:
        flags.append("no_scr_events")

    return {
        "user_id": user_id,
        "task_id": task_id,
        "condition": CONDITION_LABELS.get(task_id, task_id),
        "file": str(path),
        "n_samples": int(data.size),
        "sampling_rate_hz": fs,
        "duration_sec": duration_sec,
        "gsr_mean": float(np.nanmean(data)),
        "gsr_std": float(np.nanstd(data)),
        "gsr_min": float(np.nanmin(data)),
        "gsr_max": float(np.nanmax(data)),
        "scl_mean": float(np.nanmean(tonic)),
        "scr_count": int(len(events)),
        "scr_rate_per_min": float(len(events) / duration_sec * 60) if duration_sec else np.nan,
        "scr_amp_mean": float(np.nanmean(amplitudes)) if amplitudes.size else np.nan,
        "scr_amp_median": float(np.nanmedian(amplitudes)) if amplitudes.size else np.nan,
        "scr_amp_max": float(np.nanmax(amplitudes)) if amplitudes.size else np.nan,
        "scr_amp_sum": float(np.nansum(amplitudes)) if amplitudes.size else 0.0,
        "scr_auc_positive": float(np.trapezoid(positive_phasic, dx=1 / fs)),
        "scr_rise_time_mean_sec": float(np.nanmean(rise_times)) if rise_times.size else np.nan,
        "scr_half_recovery_mean_sec": float(np.nanmean(recoveries)) if recoveries.size else np.nan,
        "quality_flags": ";".join(flags),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SCR features from processed GSR .mat files.")
    parser.add_argument(
        "--input-dir",
        default="ICSlab_project_GSR_output/processedGSR",
        help="Directory containing per-user processed GSR .mat files.",
    )
    parser.add_argument(
        "--output",
        default="ICSlab_project_GSR_output/GSR_SCR_features.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--min-amplitude",
        type=float,
        default=0.01,
        help="Minimum SCR amplitude threshold in the same unit as the processed GSR signal.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = [summarize_file(path, args.min_amplitude) for path in sorted(input_dir.rglob("*.mat"))]
    df = pd.DataFrame(rows)
    df["user_id_num"] = pd.to_numeric(df["user_id"], errors="coerce")
    df = df.sort_values(["user_id_num", "task_id"]).drop(columns=["user_id_num"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(df)} rows to {output}")
    print(df["quality_flags"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
