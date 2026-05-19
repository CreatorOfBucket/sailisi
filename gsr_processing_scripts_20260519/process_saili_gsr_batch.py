"""
Batch entrypoint for running the project's GSR preprocessing on Saili-style data.

Expected source layout:
    <root>/T001/GSR/*.npy
    <root>/T002/GSR/*.npy
    ...

Outputs follow GSRpreprocess.process_and_save_gsr_segments:
    <output-root>/processedGSR/<subject_number>/*.mat
    <output-root>/GSR_plots/<subject_number>/*.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEUROKIT_EXTRACTED = PROJECT_ROOT / "deps_neurokit_extracted"
if (NEUROKIT_EXTRACTED / "neurokit2" / "__init__.py").is_file():
    sys.path.insert(0, str(NEUROKIT_EXTRACTED))

from GSRpreprocess import process_and_save_gsr_segments


INVALID_PROJECT_TRIGGERS = {"null", "drive"}


def subject_number(subject_name: str) -> str:
    match = re.fullmatch(r"[Tt]0*(\d+)", subject_name)
    if match:
        return match.group(1)
    if subject_name.isdigit():
        return str(int(subject_name))
    raise ValueError(f"Cannot parse numeric subject id from {subject_name!r}")


def load_and_concat_npy(files: list[Path]) -> np.ndarray:
    header = None
    parts = []

    for file_index, npy_path in enumerate(files):
        data = np.load(npy_path, allow_pickle=True)
        if data.ndim != 2:
            raise ValueError(f"{npy_path} is not a 2D array: shape={data.shape}")

        has_header = data.shape[0] > 0 and isinstance(data[0, 0], str)
        if has_header:
            current_header = data[0]
            if file_index == 0:
                header = current_header
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
    """Keep project behavior consistent with getGSRepoch.py for null/drive rows."""
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


def process_root(
    input_root: Path,
    output_root: Path,
    min_bytes: int,
    subjects: set[str] | None = None,
) -> tuple[int, int, int]:
    subjects_processed = 0
    files_processed = 0
    subjects_failed = 0

    subject_dirs = []
    for path in sorted(input_root.iterdir()):
        if not path.is_dir() or not re.fullmatch(r"[Tt]\d+", path.name):
            continue
        if subjects is not None and path.name.upper() not in subjects:
            continue
        subject_dirs.append(path)

    print(f"[INFO] input_root={input_root}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] subject_dirs={len(subject_dirs)}")

    for subject_dir in subject_dirs:
        gsr_dir = subject_dir / "GSR"
        if not gsr_dir.is_dir():
            print(f"[SKIP] {subject_dir.name}: missing GSR directory")
            continue

        npy_files = [
            path for path in sorted(gsr_dir.glob("*.npy"))
            if path.stat().st_size > min_bytes
        ]
        if not npy_files:
            print(f"[SKIP] {subject_dir.name}: no .npy larger than {min_bytes} bytes")
            continue

        try:
            sid = subject_number(subject_dir.name)
            print(f"[PROCESS] {subject_dir.name} -> {sid}: {len(npy_files)} file(s)")
            gsr_data = normalize_invalid_triggers(load_and_concat_npy(npy_files))
            process_and_save_gsr_segments(gsr_data, sid, str(output_root))
            subjects_processed += 1
            files_processed += len(npy_files)
        except Exception as exc:
            subjects_failed += 1
            print(f"[ERROR] {subject_dir.name}: {exc}")

    print("[DONE]")
    print(f"[DONE] subjects_processed={subjects_processed}")
    print(f"[DONE] files_processed={files_processed}")
    print(f"[DONE] subjects_failed={subjects_failed}")
    return subjects_processed, files_processed, subjects_failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run project GSR preprocessing on Saili data.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(r"C:\Users\26354\Desktop\saili数据处理"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <input-root>/ICSlab_project_GSR_output",
    )
    parser.add_argument("--min-bytes", type=int, default=20 * 1024)
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs to process, e.g. T001 T002.",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = (args.output_root or (input_root / "ICSlab_project_GSR_output")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = {item.upper() for item in args.subjects} if args.subjects else None
    process_root(input_root, output_root, args.min_bytes, subjects)


if __name__ == "__main__":
    main()
