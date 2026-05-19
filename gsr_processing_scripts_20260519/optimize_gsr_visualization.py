from __future__ import annotations

import argparse
import gc
import re
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_FS = 50.0
INVALID_PROJECT_TRIGGERS = {"null", "drive"}
INVALID_TRIGGERS = {"none", "no trigger", "nano", "nonenone", "base.", "hpaay", "none."}


def add_neurokit_path(project_root: Path) -> None:
    neurokit_extracted = project_root / "deps_neurokit_extracted"
    if (neurokit_extracted / "neurokit2" / "__init__.py").is_file():
        sys.path.insert(0, str(neurokit_extracted))


def subject_dir_name(subject_id: int) -> str:
    return f"T{subject_id:03d}"


def load_and_concat_npy(files: list[Path]) -> np.ndarray:
    header = None
    parts = []
    for file_index, npy_path in enumerate(files):
        data = np.load(npy_path, allow_pickle=True)
        if data.ndim != 2:
            raise ValueError(f"{npy_path} is not a 2D array: shape={data.shape}")
        has_header = data.shape[0] > 0 and isinstance(data[0, 0], str)
        if has_header:
            if file_index == 0:
                header = data[0]
            parts.append(data[1:])
        else:
            parts.append(data)
    if not parts:
        raise ValueError("No data arrays to concatenate")
    combined = np.vstack(parts)
    if header is not None:
        combined = np.vstack([header, combined])
    return combined


def normalize_invalid_triggers(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2 or data.shape[0] == 0:
        return data
    if isinstance(data[0, 0], str):
        header = [str(item) for item in data[0]]
        try:
            trigger_col = header.index("Trigger")
        except ValueError:
            return data
        first_data_row = 1
    else:
        trigger_col = 2
        first_data_row = 0

    data = data.copy()
    for row_index in range(first_data_row, data.shape[0]):
        trigger = str(data[row_index, trigger_col]).strip().lower()
        if trigger in INVALID_PROJECT_TRIGGERS:
            data[row_index, trigger_col] = "none"
    return data


def safe_trigger_name(trigger_name: str) -> str:
    safe = "".join(c for c in trigger_name if c.isalnum() or c in ("_", "-")).rstrip()
    return safe or "unknown"


def load_resampled_subject(data_root: Path, subject_id: int, min_bytes: int) -> tuple[pd.DataFrame, int]:
    gsr_dir = data_root / subject_dir_name(subject_id) / "GSR"
    if not gsr_dir.is_dir():
        raise FileNotFoundError(f"Missing GSR directory: {gsr_dir}")
    npy_files = [path for path in sorted(gsr_dir.glob("*.npy")) if path.stat().st_size > min_bytes]
    if not npy_files:
        raise FileNotFoundError(f"No .npy larger than {min_bytes} bytes in {gsr_dir}")

    data = normalize_invalid_triggers(load_and_concat_npy(npy_files))
    if isinstance(data[0, 0], str):
        columns = data[0]
        data = data[1:]
    else:
        columns = ["Timestamp", "Resistance(Koum)", "Trigger"]

    df_raw = pd.DataFrame(data, columns=columns)
    df_raw["parsed_time"] = pd.to_datetime(
        df_raw["Timestamp"].astype(str).apply(
            lambda x: x + "000" if "." in x and len(x.split(".")[-1]) < 6 else x
        ),
        format="%y-%m-%d %H:%M:%S.%f",
        errors="coerce",
    )
    df_raw["Resistance(Koum)"] = pd.to_numeric(df_raw["Resistance(Koum)"], errors="coerce")
    df_raw.dropna(subset=["parsed_time", "Resistance(Koum)"], inplace=True)
    if df_raw.empty:
        raise ValueError("Timestamp or resistance data invalid")

    df_raw = df_raw.sort_values("parsed_time").reset_index(drop=True)
    time_series = df_raw["parsed_time"]
    new_time = pd.date_range(
        start=time_series.iloc[0],
        end=time_series.iloc[-1],
        freq=pd.to_timedelta(1.0 / TARGET_FS, unit="s"),
    )

    original_seconds = (time_series - time_series.iloc[0]).dt.total_seconds().values
    new_seconds = (new_time - time_series.iloc[0]).total_seconds()
    interp_resistance = np.interp(
        new_seconds,
        original_seconds,
        df_raw["Resistance(Koum)"].astype(float).values,
    )
    trigger_idx = np.searchsorted(original_seconds, new_seconds, side="left")
    trigger_idx = np.clip(trigger_idx, 0, len(df_raw) - 1)
    interp_trigger = df_raw["Trigger"].iloc[trigger_idx].values

    df_full = pd.DataFrame(
        {
            "Timestamp": new_time.strftime("%y-%m-%d %H:%M:%S.%f").str[:-3],
            "Resistance(Koum)": interp_resistance,
            "Trigger": interp_trigger,
            "parsed_time": new_time,
        }
    )

    timestamps_ns = df_full["parsed_time"].astype("datetime64[ns]").astype(np.int64)
    time_diffs = np.diff(timestamps_ns / 1e9)
    valid_diffs = time_diffs[time_diffs > 0]
    if len(valid_diffs) == 0:
        raise ValueError("Cannot calculate valid time interval")
    sampling_rate = int(round(1 / np.median(valid_diffs)))
    if sampling_rate < 2:
        raise ValueError(f"Sampling rate too low: {sampling_rate} Hz")

    trigger_corrections = {"anxitey": "anxiety", "terrify": "disgust"}
    df_full["Trigger"] = df_full["Trigger"].astype(str).str.lower().str.strip()
    df_full["Trigger"] = df_full["Trigger"].replace(trigger_corrections)
    return df_full, sampling_rate


def existing_targets(plot_root: Path, subject_id: int) -> set[str]:
    folder = plot_root / str(subject_id)
    if not folder.is_dir():
        return set()
    pattern = re.compile(rf"^{subject_id}_(.+)_plot\.png$", re.IGNORECASE)
    targets = set()
    for path in folder.glob("*_plot.png"):
        match = pattern.match(path.name)
        if match:
            targets.add(match.group(1).lower())
    return targets


def index_array(info: dict, key: str, length: int) -> np.ndarray:
    values = np.asarray(info.get(key, []), dtype=float)
    values = values[np.isfinite(values)]
    values = values.astype(int)
    return values[(values >= 0) & (values < length)]


def padded_ylim(values: np.ndarray, lower_q: float = 0.5, upper_q: float = 99.5, pad: float = 0.08):
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(values, [lower_q, upper_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(values), np.nanmax(values)
    span = max(hi - lo, 1e-6)
    return lo - span * pad, hi + span * pad


def plot_clear(plot_root: Path, subject_id: int, task: str, signals, info: dict, sampling_rate: int) -> Path:
    length = len(signals)
    x = np.arange(length, dtype=float) / sampling_rate
    onsets = index_array(info, "SCR_Onsets", length)
    peaks = index_array(info, "SCR_Peaks", length)
    recovery = index_array(info, "SCR_Recovery", length)

    fig = plt.figure(figsize=(12.0, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 1.25, 0.48, 1.05])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)

    fig.suptitle(f"Electrodermal Activity (EDA) - User {subject_id}, {task.upper()}", fontweight="bold")

    ax1.plot(x, signals["EDA_Raw"], color="#AEB8C2", lw=0.7, alpha=0.65, label="Raw")
    ax1.plot(x, signals["EDA_Clean"], color="#B517C9", lw=1.15, label="Cleaned")
    ax1.set_title("Raw and Cleaned Signal", pad=7)
    ax1.set_ylabel("uS")
    ax1.set_ylim(*padded_ylim(np.r_[signals["EDA_Raw"].values, signals["EDA_Clean"].values]))
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2.axhline(0, color="#98A2AD", lw=0.8, alpha=0.8)
    ax2.plot(x, signals["EDA_Phasic"], color="#1F77B4", lw=0.95, label="Phasic Component")
    if len(peaks):
        phasic = signals["EDA_Phasic"].to_numpy()
        ax2.scatter(x[peaks], phasic[peaks], s=9, color="#F5A623", alpha=0.75, label="SCR Peaks", zorder=3)
    ax2.set_title("Skin Conductance Response (SCR)", pad=7)
    ax2.set_ylabel("uS")
    ax2.set_ylim(*padded_ylim(signals["EDA_Phasic"].values, pad=0.12))
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.text(
        0.01,
        0.95,
        f"NeuroKit SCR peaks: {len(peaks)}",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D0D7DE", alpha=0.9),
    )

    events = [
        (onsets, 2, "#2CA02C", "SCR Onsets"),
        (peaks, 1, "#F5A623", "SCR Peaks"),
        (recovery, 0, "#7E57C2", "SCR Half Recovery"),
    ]
    for indexes, row, color, label in events:
        if len(indexes):
            ax3.scatter(x[indexes], np.full(len(indexes), row), s=55, marker="|", color=color, alpha=0.72, label=label)
    ax3.set_title("SCR Event Markers", pad=6)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["Half recovery", "Peaks", "Onsets"])
    ax3.set_ylim(-0.6, 2.6)
    ax3.grid(True, axis="x", color="#E6EAEE", lw=0.7)
    ax3.grid(False, axis="y")
    ax3.legend(loc="center left", bbox_to_anchor=(1.004, 0.5), frameon=False, borderaxespad=0)

    ax4.plot(x, signals["EDA_Tonic"], color="#6F3CC3", lw=1.3, label="Tonic Component")
    ax4.set_title("Skin Conductance Level (SCL)", pad=7)
    ax4.set_ylabel("uS")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_ylim(*padded_ylim(signals["EDA_Tonic"].values, pad=0.10))
    ax4.legend(loc="upper right", framealpha=0.9)

    for ax in (ax1, ax2, ax4):
        ax.grid(True, color="#E6EAEE", lw=0.7)
    for ax in (ax1, ax2, ax3, ax4):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0)

    out = plot_root / str(subject_id) / f"{subject_id}_{task}_plot.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def process_subject(data_root: Path, plot_root: Path, subject_id: int, min_bytes: int, nk) -> tuple[int, list[str]]:
    targets = existing_targets(plot_root, subject_id)
    if not targets:
        return 0, []

    df_full, sampling_rate = load_resampled_subject(data_root, subject_id, min_bytes)
    df_valid = df_full[
        df_full["Trigger"].notna()
        & (df_full["Trigger"] != "")
        & (~df_full["Trigger"].isin(INVALID_TRIGGERS))
    ].copy()
    if df_valid.empty:
        return 0, [f"{subject_id}: no valid trigger segments"]

    df_valid["trigger_group"] = (df_valid["Trigger"] != df_valid["Trigger"].shift()).cumsum()
    segments = df_valid.groupby(["trigger_group", "Trigger"]).agg(
        start_time=("parsed_time", "first"),
        end_time=("parsed_time", "last"),
    ).reset_index()

    updated = set()
    notes = []
    for _, row in segments.iterrows():
        trigger_name = str(row["Trigger"])
        task = safe_trigger_name(trigger_name)
        if task not in targets:
            continue

        segment_df = df_full[
            (df_full["parsed_time"] >= row["start_time"])
            & (df_full["parsed_time"] <= row["end_time"])
        ].copy()
        if len(segment_df) < max(2 * sampling_rate, 10):
            notes.append(f"{subject_id}_{task}: too short ({len(segment_df)} points)")
            continue

        resistance = segment_df["Resistance(Koum)"].astype(float).values
        resistance[resistance == 0] = 1e-9
        conductance = 1000.0 / resistance
        try:
            signals, info = nk.eda_process(conductance, sampling_rate=sampling_rate)
            plot_clear(plot_root, subject_id, task, signals, info, sampling_rate)
            updated.add(task)
        except Exception as exc:
            notes.append(f"{subject_id}_{task}: {exc}")

    missing = sorted(targets - updated)
    if missing:
        notes.append(f"{subject_id}: not updated {missing}")
    return len(updated), notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate clear NeuroKit-based GSR visualizations.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing T001/GSR/*.npy style folders.")
    parser.add_argument("--plot-root", type=Path, required=True, help="Existing GSR_plots directory to overwrite.")
    parser.add_argument("--project-root", type=Path, required=True, help="Project root containing deps_neurokit_extracted.")
    parser.add_argument("--subject-start", type=int, default=1)
    parser.add_argument("--subject-end", type=int, default=999)
    parser.add_argument("--min-bytes", type=int, default=20 * 1024)
    args = parser.parse_args()

    add_neurokit_path(args.project_root)
    import neurokit2 as nk

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    subjects = sorted(
        int(path.name)
        for path in args.plot_root.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and args.subject_start <= int(path.name) <= args.subject_end
    )
    print(f"[START] subjects={len(subjects)}")

    total_updated = 0
    all_notes: list[str] = []
    for index, subject_id in enumerate(subjects, start=1):
        try:
            updated, notes = process_subject(args.data_root, args.plot_root, subject_id, args.min_bytes, nk)
            total_updated += updated
            all_notes.extend(notes)
            print(f"[SUBJECT {index}/{len(subjects)}] {subject_id}: updated={updated}")
        except Exception as exc:
            all_notes.append(f"{subject_id}: fatal {exc}")
            print(f"[SUBJECT {index}/{len(subjects)}] {subject_id}: ERROR {exc}")
        finally:
            plt.close("all")
            gc.collect()

    print(f"[DONE] updated_files={total_updated}")
    if all_notes:
        print("[NOTES]")
        for note in all_notes:
            print(note)


if __name__ == "__main__":
    main()
