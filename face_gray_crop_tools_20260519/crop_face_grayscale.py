from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image

try:
    LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
    if LOCAL_DEPS.is_dir():
        sys.path.insert(0, str(LOCAL_DEPS))
    import cv2
except Exception:  # pragma: no cover - fallback for minimal environments
    cv2 = None


TRUST_SUFFIX_RE = re.compile(r"T(?:12|22|32)$", re.IGNORECASE)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int
    method: str


def trust_label(path: Path) -> str:
    match = TRUST_SUFFIX_RE.search(path.stem)
    return match.group(0).upper() if match else "UNKNOWN"


def clamp_box(left: float, top: float, width: float, height: float, img_w: int, img_h: int, method: str) -> CropBox:
    left_i = int(round(max(0, min(left, img_w - 1))))
    top_i = int(round(max(0, min(top, img_h - 1))))
    right_i = int(round(max(left_i + 1, min(left + width, img_w))))
    bottom_i = int(round(max(top_i + 1, min(top + height, img_h))))
    return CropBox(left_i, top_i, right_i, bottom_i, method)


def fallback_box(img_w: int, img_h: int) -> CropBox:
    crop_w = img_w * 0.56
    crop_h = img_h * 0.92
    left = img_w * 0.50 - crop_w / 2
    top = img_h * 0.08
    return clamp_box(left, top, crop_w, crop_h, img_w, img_h, "fallback_complete_visible_face")


def expand_for_complete_visible_face(box: CropBox, img_w: int, img_h: int) -> CropBox:
    min_w = img_w * 0.42
    min_h = img_h * 0.68
    width = box.right - box.left
    height = box.bottom - box.top
    cx = (box.left + box.right) / 2
    cy = (box.top + box.bottom) / 2

    width = max(width, min_w)
    height = max(height, min_h)
    left = cx - width / 2
    top = cy - height * 0.40

    # In this dataset many participants sit low in frame. Force enough room
    # below the detected face so the visible chin/mouth is not cropped away.
    bottom = max(top + height, img_h * 0.90)
    if bottom > img_h:
        top -= bottom - img_h
        bottom = img_h

    expanded = clamp_box(left, top, width, bottom - top, img_w, img_h, box.method)
    if expanded.bottom < int(img_h * 0.95):
        expanded = clamp_box(expanded.left, expanded.top, expanded.right - expanded.left, img_h - expanded.top, img_w, img_h, box.method)
    return expanded


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)

    ycbcr_skin = (y > 45) & (cb >= 72) & (cb <= 140) & (cr >= 120) & (cr <= 185)
    rgb_skin = (r > 55) & (g > 30) & (b > 20) & ((maxc - minc) > 10) & (r >= b * 0.95) & (g >= b * 0.75)
    return ycbcr_skin & rgb_skin


def get_cascade_path(name: str) -> str | None:
    if cv2 is None:
        return None

    src = Path(cv2.data.haarcascades) / name
    if not src.exists():
        return None

    # OpenCV on Windows can fail to open XMLs from non-ASCII paths.
    dst = Path(tempfile.gettempdir()) / name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)
    return str(dst)


def detect_cv_face_box(rgb: np.ndarray) -> CropBox | None:
    if cv2 is None:
        return None

    cascade_path = get_cascade_path("haarcascade_frontalface_alt2.xml")
    if cascade_path is None:
        return None

    img_h, img_w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(65, 65),
        maxSize=(380, 380),
    )
    if len(faces) == 0:
        return None

    best = None
    best_score = None
    for x, y, w, h in faces:
        cx = x + w / 2
        cy = y + h / 2
        if not (img_w * 0.34 <= cx <= img_w * 0.74 and img_h * 0.12 <= cy <= img_h * 0.78):
            continue
        area = w * h
        center_penalty = abs(cx - img_w * 0.50) * 90.0 + abs(cy - img_h * 0.45) * 35.0
        score = area - center_penalty
        if best_score is None or score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return None

    x, y, w, h = best
    crop_w = max(390.0, min(650.0, w * 2.75))
    crop_h = max(470.0, min(680.0, h * 3.35))
    cx = x + w / 2
    cy = y + h / 2
    left = cx - crop_w / 2
    top = cy - crop_h * 0.40
    return expand_for_complete_visible_face(clamp_box(left, top, crop_w, crop_h, img_w, img_h, "opencv_haar"), img_w, img_h)


def detect_skin_box(rgb: np.ndarray) -> CropBox | None:
    if cv2 is None:
        return None

    img_h, img_w = rgb.shape[:2]
    x0 = int(img_w * 0.28)
    x1 = int(img_w * 0.74)
    y0 = int(img_h * 0.05)
    y1 = int(img_h * 0.84)

    mask = np.zeros((img_h, img_w), dtype=bool)
    mask[y0:y1, x0:x1] = skin_mask(rgb[y0:y1, x0:x1])

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((7, 7), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    best = None
    best_score = None

    for label in range(1, num_labels):
        min_c = int(stats[label, cv2.CC_STAT_LEFT])
        min_r = int(stats[label, cv2.CC_STAT_TOP])
        box_w = int(stats[label, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 900:
            continue

        max_c = min_c + box_w
        max_r = min_r + box_h
        if box_w < 25 or box_h < 25 or box_w > img_w * 0.35 or box_h > img_h * 0.55:
            continue

        cx, cy = centroids[label]
        if not (img_w * 0.32 <= cx <= img_w * 0.72 and img_h * 0.10 <= cy <= img_h * 0.78):
            continue

        center_penalty = abs(cx - img_w * 0.50) * 1.0 + abs(cy - img_h * 0.46) * 0.35
        score = area - center_penalty
        if best_score is None or score > best_score:
            best_score = score
            best = (min_c, min_r, max_c, max_r, cx, cy)

    if best is None:
        return None

    min_c, min_r, max_c, max_r, cx, cy = best
    skin_w = max_c - min_c
    skin_h = max_r - min_r

    crop_w = max(330.0, min(560.0, skin_w * 2.35))
    crop_h = max(390.0, min(610.0, skin_h * 2.45))
    left = cx - crop_w / 2
    top = cy - crop_h * 0.57
    return clamp_box(left, top, crop_w, crop_h, img_w, img_h, "skin_region")


def choose_crop_box(image: Image.Image) -> CropBox:
    rgb = np.asarray(image.convert("RGB"))
    box = detect_cv_face_box(rgb)
    if box is not None:
        return box
    return fallback_box(image.width, image.height)


def iter_target_images(root: Path, start: int, end: int):
    for idx in range(start, end + 1):
        subject = f"T{idx:03d}"
        image_dir = root / subject / "Image"
        if not image_dir.is_dir():
            yield subject, None, "missing_image_dir"
            continue
        files = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and TRUST_SUFFIX_RE.search(p.stem)
        )
        if not files:
            yield subject, None, "no_matching_images"
            continue
        for path in files:
            yield subject, path, "ok"


def build_groups(root: Path, start: int, end: int):
    groups: dict[tuple[str, str], list[Path]] = {}
    notes = []

    for idx in range(start, end + 1):
        subject = f"T{idx:03d}"
        image_dir = root / subject / "Image"
        if not image_dir.is_dir():
            notes.append((subject, "missing_image_dir"))
            continue
        files = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and TRUST_SUFFIX_RE.search(p.stem)
        )
        if not files:
            notes.append((subject, "no_matching_images"))
            continue
        for path in files:
            groups.setdefault((subject, trust_label(path)), []).append(path)

    return groups, notes


def sample_files(files: list[Path], count: int = 15) -> list[Path]:
    if len(files) <= count:
        return files
    indexes = np.linspace(0, len(files) - 1, num=count, dtype=int)
    return [files[int(i)] for i in indexes]


def median_group_box(files: list[Path]) -> CropBox:
    boxes: list[CropBox] = []
    for path in sample_files(files):
        try:
            with Image.open(path) as im:
                boxes.append(choose_crop_box(im))
        except Exception:
            continue

    if not boxes:
        with Image.open(files[0]) as im:
            return fallback_box(im.width, im.height)

    opencv_boxes = [b for b in boxes if b.method == "opencv_haar"]
    if opencv_boxes:
        if len(opencv_boxes) >= 3:
            boxes = opencv_boxes

    arr = np.array([[b.left, b.top, b.right, b.bottom] for b in boxes], dtype=np.float32)
    left, top, right, bottom = np.median(arr, axis=0).tolist()
    method = "group_" + "+".join(sorted({b.method for b in boxes}))
    with Image.open(files[0]) as im:
        return clamp_box(left, top, right - left, bottom - top, im.width, im.height, method)


def crop_one(task):
    src_s, out_s, box_values, quality = task
    left, top, right, bottom = box_values
    try:
        with Image.open(src_s) as im:
            gray = im.convert("L").crop((left, top, right, bottom))
            gray.save(out_s, quality=quality)
        return True, src_s, out_s, ""
    except Exception as exc:
        return False, src_s, out_s, repr(exc)


def process(root: Path, output: Path, start: int, end: int, quality: int, workers: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    range_tag = f"T{start:03d}_T{end:03d}"
    group_manifest_path = output / f"group_crop_boxes_{range_tag}.csv"
    summary_path = output / f"crop_summary_{range_tag}.csv"
    failures_path = output / f"failed_images_{range_tag}.csv"

    total = 0
    failed = 0
    groups, skipped_subject_notes = build_groups(root, start, end)

    group_boxes: dict[tuple[str, str], CropBox] = {}
    with group_manifest_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "trust_label", "image_count", "left", "top", "right", "bottom", "method"])
        for key, files in sorted(groups.items()):
            box = median_group_box(files)
            group_boxes[key] = box
            writer.writerow([key[0], key[1], len(files), box.left, box.top, box.right, box.bottom, box.method])

    with summary_path.open("w", newline="", encoding="utf-8-sig") as summary_fh:
        summary_writer = csv.writer(summary_fh)
        summary_writer.writerow(["subject", "trust_label", "image_count", "output_dir", "status"])

        tasks = []
        for (subject, _label), files in sorted(groups.items()):
            out_dir = output / subject
            out_dir.mkdir(parents=True, exist_ok=True)
            box = group_boxes[(subject, _label)]
            box_values = (box.left, box.top, box.right, box.bottom)
            for src in files:
                tasks.append((str(src), str(out_dir / src.name), box_values, quality))
            summary_writer.writerow([subject, _label, len(files), str(out_dir), "queued"])

        for subject, status in skipped_subject_notes:
            summary_writer.writerow([subject, "", 0, "", status])

    with failures_path.open("w", newline="", encoding="utf-8-sig") as fail_fh:
        fail_writer = csv.writer(fail_fh)
        fail_writer.writerow(["source", "output", "message"])

        if workers <= 1:
            iterator = map(crop_one, tasks)
            for ok, src_s, out_s, message in iterator:
                if ok:
                    total += 1
                else:
                    failed += 1
                    fail_writer.writerow([src_s, out_s, message])
                if total and total % 10000 == 0:
                    print(f"processed={total} failed={failed}", flush=True)
        else:
            with Pool(processes=workers) as pool:
                for ok, src_s, out_s, message in pool.imap_unordered(crop_one, tasks, chunksize=96):
                    if ok:
                        total += 1
                    else:
                        failed += 1
                        fail_writer.writerow([src_s, out_s, message])
                    if total and total % 10000 == 0:
                        print(f"processed={total} failed={failed}", flush=True)

    print(f"done processed={total} failed={failed} summary={summary_path} group_boxes={group_manifest_path} failures={failures_path}")
    if skipped_subject_notes:
        print("subject_notes=" + "; ".join(f"{s}:{note}" for s, note in skipped_subject_notes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop grayscale face regions for trust-labeled images.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=158)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (cpu_count() or 2) - 1)))
    args = parser.parse_args()
    process(args.root, args.output, args.start, args.end, args.quality, args.workers)


if __name__ == "__main__":
    main()
