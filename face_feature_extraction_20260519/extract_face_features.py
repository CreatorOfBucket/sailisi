"""
Extract EAR (Eye Aspect Ratio), Face3D features, and AU (Action Units)
from cropped face grayscale images using MediaPipe Face Landmarker (Tasks API).

Input:  face_gray_T001_T158_T12_T22_T32_20260519_complete_visible/
Output: face_features_output/
  - face_features.csv         (per-image features)
  - face_features_summary.csv (per-subject per-condition summary)
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# MediaPipe Tasks API (0.10.x)
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe import Image as MpImage
from mediapipe import ImageFormat

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\26354\Desktop\saili数据处理")
INPUT_DIR = BASE_DIR / "face_gray_T001_T158_T12_T22_T32_20260519_complete_visible"
OUTPUT_DIR = BASE_DIR / "face_features_output"
FEATURES_CSV = OUTPUT_DIR / "face_features.csv"
SUMMARY_CSV = OUTPUT_DIR / "face_features_summary.csv"
FAILED_LOG = OUTPUT_DIR / "failed_images.csv"
MODEL_PATH = Path("C:/Users/26354/face_landmarker_v2.task")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices (478-landmark model)
# ---------------------------------------------------------------------------

# --- Eyes ---
# Right eye (subject's left, observer's right)
RIGHT_EYE_CORNER_INNER = 133
RIGHT_EYE_CORNER_OUTER = 33
RIGHT_EYE_UPPER1 = 159
RIGHT_EYE_UPPER2 = 160
RIGHT_EYE_LOWER1 = 145
RIGHT_EYE_LOWER2 = 144
RIGHT_EYE_INDICES = [33, 133, 157, 158, 159, 160, 161, 173]

# Left eye (subject's right, observer's left)
LEFT_EYE_CORNER_INNER = 362
LEFT_EYE_CORNER_OUTER = 263
LEFT_EYE_UPPER1 = 386
LEFT_EYE_UPPER2 = 385
LEFT_EYE_LOWER1 = 374
LEFT_EYE_LOWER2 = 373
LEFT_EYE_INDICES = [362, 263, 384, 385, 386, 387, 388, 398]

# --- Eyebrows ---
RIGHT_EYEBROW_INNER = 107
RIGHT_EYEBROW_OUTER = 55
RIGHT_EYEBROW_MID = 66
LEFT_EYEBROW_INNER = 336
LEFT_EYEBROW_OUTER = 285
LEFT_EYEBROW_MID = 296

# --- Nose ---
NOSE_TIP = 4
NOSE_BRIDGE = 6
NOSE_BASE = 2
NOSE_BOTTOM_L = 98
NOSE_BOTTOM_R = 327

# --- Mouth ---
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_UPPER = 13
MOUTH_LOWER = 14

# --- Face contour ---
CHIN = 152
FOREHEAD = 10
FACE_LEFT = 234
FACE_RIGHT = 454

# --- Jaw ---
JAW_LEFT = 172
JAW_RIGHT = 397


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _get_pt(landmarks, idx):
    """Get (x, y, z) from a landmarks list at index idx."""
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])


def _get_pt_2d(landmarks, idx, img_w, img_h):
    """Get pixel (x, y) from a landmarks list at index idx."""
    lm = landmarks[idx]
    return np.array([lm.x * img_w, lm.y * img_h])


# ---------------------------------------------------------------------------
# EAR: Eye Aspect Ratio
# ---------------------------------------------------------------------------

def compute_ear(landmarks, img_w, img_h):
    """
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Uses pixel coordinates (_get_pt_2d) to preserve aspect ratio on non-square images.
    """
    def eye_ear(i_corner_in, i_upper1, i_upper2, i_corner_out, i_lower1, i_lower2):
        p1 = _get_pt_2d(landmarks, i_corner_in, img_w, img_h)
        p2 = _get_pt_2d(landmarks, i_upper1, img_w, img_h)
        p3 = _get_pt_2d(landmarks, i_upper2, img_w, img_h)
        p4 = _get_pt_2d(landmarks, i_corner_out, img_w, img_h)
        p5 = _get_pt_2d(landmarks, i_lower1, img_w, img_h)
        p6 = _get_pt_2d(landmarks, i_lower2, img_w, img_h)
        v = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        h = 2.0 * np.linalg.norm(p1 - p4)
        return v / h if h > 1e-9 else np.nan

    r_ear = eye_ear(RIGHT_EYE_CORNER_INNER, RIGHT_EYE_UPPER1, RIGHT_EYE_UPPER2,
                    RIGHT_EYE_CORNER_OUTER, RIGHT_EYE_LOWER1, RIGHT_EYE_LOWER2)
    l_ear = eye_ear(LEFT_EYE_CORNER_INNER, LEFT_EYE_UPPER1, LEFT_EYE_UPPER2,
                    LEFT_EYE_CORNER_OUTER, LEFT_EYE_LOWER1, LEFT_EYE_LOWER2)
    return {
        "ear_right": r_ear,
        "ear_left": l_ear,
        "ear_avg": np.nanmean([r_ear, l_ear]),
    }


# ---------------------------------------------------------------------------
# Head pose (yaw, pitch, roll)
# ---------------------------------------------------------------------------

def compute_head_pose(landmarks, img_w, img_h):
    image_pts = np.array([
        _get_pt_2d(landmarks, NOSE_TIP, img_w, img_h),
        _get_pt_2d(landmarks, CHIN, img_w, img_h),
        _get_pt_2d(landmarks, LEFT_EYE_CORNER_INNER, img_w, img_h),
        _get_pt_2d(landmarks, RIGHT_EYE_CORNER_INNER, img_w, img_h),
        _get_pt_2d(landmarks, MOUTH_LEFT, img_w, img_h),
        _get_pt_2d(landmarks, MOUTH_RIGHT, img_w, img_h),
    ], dtype=np.float32)

    model_pts = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-35.0, 20.0, -20.0),
        (35.0, 20.0, -20.0),
        (-28.0, -32.0, -5.0),
        (28.0, -32.0, -5.0),
    ], dtype=np.float32)

    fl = img_w
    cam = np.array([[fl, 0, img_w / 2], [0, fl, img_h / 2], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)

    ok, rvec, _ = cv2.solvePnP(model_pts, image_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {"yaw_deg": np.nan, "pitch_deg": np.nan, "roll_deg": np.nan}

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
        roll = np.arctan2(rmat[2, 1], rmat[2, 2])
    else:
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = np.arctan2(-rmat[0, 1], rmat[1, 1])
        roll = 0.0

    return {
        "yaw_deg": float(np.degrees(yaw)),
        "pitch_deg": float(np.degrees(pitch)),
        "roll_deg": float(np.degrees(roll)),
    }


# ---------------------------------------------------------------------------
# Face3D features
# ---------------------------------------------------------------------------

def compute_face3d_features(landmarks, img_w, img_h):
    # Use pixel coordinates for all 2D measurements (preserves aspect ratio)
    def _pt2(idx):
        return _get_pt_2d(landmarks, idx, img_w, img_h)

    def _pt3(idx):
        """Returns (pixel_x, pixel_y, z_metric)."""
        p2 = _pt2(idx)
        return np.array([p2[0], p2[1], landmarks[idx].z])

    left_eye_center = np.mean([_pt3(i) for i in (LEFT_EYE_CORNER_INNER, LEFT_EYE_CORNER_OUTER)], axis=0)
    right_eye_center = np.mean([_pt3(i) for i in (RIGHT_EYE_CORNER_INNER, RIGHT_EYE_CORNER_OUTER)], axis=0)
    # IED in pixel space (ignoring z for normalization)
    ied = np.linalg.norm(left_eye_center[:2] - right_eye_center[:2])
    if ied < 1e-9:
        ied = 1.0

    nose = _pt3(NOSE_TIP)
    chin = _pt3(CHIN)
    forehead = _pt3(FOREHEAD)
    fl = _pt3(FACE_LEFT)
    fr = _pt3(FACE_RIGHT)
    m_lt = _pt3(MOUTH_LEFT)
    m_rt = _pt3(MOUTH_RIGHT)
    m_up = _pt3(MOUTH_UPPER)
    m_lo = _pt3(MOUTH_LOWER)
    j_lt = _pt3(JAW_LEFT)
    j_rt = _pt3(JAW_RIGHT)
    n_lt = _pt3(NOSE_BOTTOM_L)
    n_rt = _pt3(NOSE_BOTTOM_R)

    # Nose protrusion (z-depth relative to face plane, z is metric)
    face_plane_z = (forehead[2] + chin[2]) / 2.0
    nose_protrusion = nose[2] - face_plane_z

    # Eye sizes (pixel space, then normalized by pixel IED)
    def _eye_size(indices):
        pts = np.array([_pt3(i) for i in indices])
        return (np.max(pts[:, 1]) - np.min(pts[:, 1])) / ied, (np.max(pts[:, 0]) - np.min(pts[:, 0])) / ied

    r_eh, r_ew = _eye_size(RIGHT_EYE_INDICES)
    l_eh, l_ew = _eye_size(LEFT_EYE_INDICES)

    def _dist2d_3d(a, b):
        """2D pixel distance between two 3D points."""
        return np.linalg.norm(a[:2] - b[:2])

    return {
        "face3d_width_ratio": float(_dist2d_3d(fr, fl) / ied),
        "face3d_height_ratio": float(_dist2d_3d(forehead, chin) / ied),
        "face3d_nose_depth": float(nose[2]),
        "face3d_nose_protrusion": float(nose_protrusion),
        "face3d_nose_to_chin_ratio": float(_dist2d_3d(nose, chin) / ied),
        "face3d_nose_to_forehead_ratio": float(_dist2d_3d(nose, forehead) / ied),
        "face3d_mouth_width_ratio": float(_dist2d_3d(m_rt, m_lt) / ied),
        "face3d_mouth_height_ratio": float(_dist2d_3d(m_up, m_lo) / ied),
        "face3d_jaw_width_ratio": float(_dist2d_3d(j_rt, j_lt) / ied),
        "face3d_nose_width_ratio": float(_dist2d_3d(n_rt, n_lt) / ied),
        "face3d_nose_offset": float(nose[0] - (fl[0] + fr[0]) / 2.0),
        "face3d_left_eye_h": float(l_eh),
        "face3d_left_eye_w": float(l_ew),
        "face3d_right_eye_h": float(r_eh),
        "face3d_right_eye_w": float(r_ew),
        "face3d_inter_eye_dist_norm": float(ied),
        "face3d_eye_z_left": float(np.mean([landmarks[i].z for i in LEFT_EYE_INDICES])),
        "face3d_eye_z_right": float(np.mean([landmarks[i].z for i in RIGHT_EYE_INDICES])),
    }


# ---------------------------------------------------------------------------
# AU: Action Units
# ---------------------------------------------------------------------------

def compute_au_features(landmarks, img_w, img_h):
    # Normalization factor: inter-eye distance in pixel space
    le_c = np.mean([_get_pt_2d(landmarks, LEFT_EYE_CORNER_INNER, img_w, img_h),
                     _get_pt_2d(landmarks, LEFT_EYE_CORNER_OUTER, img_w, img_h)], axis=0)
    re_c = np.mean([_get_pt_2d(landmarks, RIGHT_EYE_CORNER_INNER, img_w, img_h),
                     _get_pt_2d(landmarks, RIGHT_EYE_CORNER_OUTER, img_w, img_h)], axis=0)
    norm = np.linalg.norm(le_c - re_c)
    if norm < 1e-9:
        norm = 1.0

    def _dist2d(i1, i2):
        p1 = _get_pt_2d(landmarks, i1, img_w, img_h)
        p2 = _get_pt_2d(landmarks, i2, img_w, img_h)
        return np.linalg.norm(p1 - p2)

    def _dist2d_pt(pt1, i2):
        p2 = _get_pt_2d(landmarks, i2, img_w, img_h)
        return np.linalg.norm(pt1 - p2)

    nose_base_pt = _get_pt_2d(landmarks, NOSE_BASE, img_w, img_h)
    mouth_up_pt = _get_pt_2d(landmarks, MOUTH_UPPER, img_w, img_h)
    mouth_lo_pt = _get_pt_2d(landmarks, MOUTH_LOWER, img_w, img_h)
    mouth_lt_pt = _get_pt_2d(landmarks, MOUTH_LEFT, img_w, img_h)
    mouth_rt_pt = _get_pt_2d(landmarks, MOUTH_RIGHT, img_w, img_h)
    mouth_c = (mouth_lt_pt + mouth_rt_pt) / 2.0
    chin_pt = _get_pt_2d(landmarks, CHIN, img_w, img_h)

    # Eye opening heights
    r_eye_h = _dist2d(RIGHT_EYE_UPPER1, RIGHT_EYE_LOWER1) / norm
    l_eye_h = _dist2d(LEFT_EYE_UPPER1, LEFT_EYE_LOWER1) / norm
    r_eye_w = _dist2d(RIGHT_EYE_CORNER_INNER, RIGHT_EYE_CORNER_OUTER) / norm
    l_eye_w = _dist2d(LEFT_EYE_CORNER_INNER, LEFT_EYE_CORNER_OUTER) / norm

    mouth_w = _dist2d(MOUTH_LEFT, MOUTH_RIGHT) / norm

    # Jaw
    j_lt = _get_pt_2d(landmarks, JAW_LEFT, img_w, img_h)
    j_rt = _get_pt_2d(landmarks, JAW_RIGHT, img_w, img_h)
    jaw_w = np.linalg.norm(j_rt - j_lt)

    return {
        # AU1: Inner Brow Raiser
        "au1_inner_brow_raise_r": float(_dist2d(RIGHT_EYEBROW_INNER, RIGHT_EYE_CORNER_INNER) / norm),
        "au1_inner_brow_raise_l": float(_dist2d(LEFT_EYEBROW_INNER, LEFT_EYE_CORNER_INNER) / norm),
        # AU2: Outer Brow Raiser
        "au2_outer_brow_raise_r": float(_dist2d(RIGHT_EYEBROW_OUTER, RIGHT_EYE_CORNER_OUTER) / norm),
        "au2_outer_brow_raise_l": float(_dist2d(LEFT_EYEBROW_OUTER, LEFT_EYE_CORNER_OUTER) / norm),
        # AU4: Brow Lowerer
        "au4_brow_lower_r": float(_dist2d_pt(re_c, RIGHT_EYEBROW_MID) / norm),
        "au4_brow_lower_l": float(_dist2d_pt(le_c, LEFT_EYEBROW_MID) / norm),
        # AU6: Cheek Raiser (eye narrowing)
        "au6_cheek_raise_r": float(r_eye_h),
        "au6_cheek_raise_l": float(l_eye_h),
        # AU7: Lid Tightener
        "au7_lid_tighten_r": float(r_eye_h),
        "au7_lid_tighten_l": float(l_eye_h),
        # AU9: Nose Wrinkle
        "au9_nose_wrinkle": float(_dist2d(NOSE_BOTTOM_L, NOSE_BOTTOM_R) / norm),
        # AU10: Upper Lip Raiser
        "au10_upper_lip_raise": float(np.linalg.norm(mouth_up_pt - nose_base_pt) / norm),
        # AU12: Lip Corner Puller (smile)
        "au12_lip_corner_pull": float(mouth_w),
        # AU15: Lip Corner Depressor
        "au15_lip_corner_depress_r": float((mouth_lt_pt[1] - mouth_c[1]) / norm),
        "au15_lip_corner_depress_l": float((mouth_rt_pt[1] - mouth_c[1]) / norm),
        # AU17: Chin Raiser
        "au17_chin_raise": float(np.linalg.norm(mouth_lo_pt - chin_pt) / norm),
        # AU20: Lip Stretch
        "au20_lip_stretch": float(mouth_w * norm / jaw_w if jaw_w > 0 else mouth_w),
        # AU25: Lips Part
        "au25_lips_part": float(_dist2d(MOUTH_UPPER, MOUTH_LOWER) / norm),
        # AU26: Jaw Drop
        "au26_jaw_drop": float(np.linalg.norm(mouth_lo_pt - chin_pt) / norm),
        # AU43: Eye Closure
        "au43_eye_closure_r": float(r_eye_h * r_eye_w),
        "au43_eye_closure_l": float(l_eye_h * l_eye_w),
        # Composite smile index
        "au_smile_index": float(mouth_w),
    }


# ---------------------------------------------------------------------------
# Single image processing
# ---------------------------------------------------------------------------

def imread_unicode(path):
    """Read image with unicode path support (cv2.imread fails on Chinese paths)."""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def process_image(image_path, landmarker, verbose=False):
    img_bgr = imread_unicode(str(image_path))
    if img_bgr is None:
        return None

    img_h, img_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = MpImage(image_format=ImageFormat.SRGB, data=img_rgb)

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    try:
        ear = compute_ear(landmarks, img_w, img_h)
        pose = compute_head_pose(landmarks, img_w, img_h)
        f3d = compute_face3d_features(landmarks, img_w, img_h)
        au = compute_au_features(landmarks, img_w, img_h)
    except Exception:
        if verbose:
            traceback.print_exc()
        return None

    parts = image_path.stem.split("_")
    subj = parts[0] if parts else ""
    task = parts[-1] if len(parts) > 1 else ""

    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "subject_id": subj,
        "task_id": task,
        "condition": {"T12": "low_trust", "T22": "medium_trust", "T32": "high_trust"}.get(task, "unknown"),
        "face_detected": 1,
        **ear,
        **pose,
        **f3d,
        **au,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHUNK_SIZE = 50000  # flush to CSV every N images


def save_chunk(rows, is_first):
    df_chunk = pd.DataFrame(rows)
    if not is_first:
        df_chunk.to_csv(FEATURES_CSV, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        df_chunk.to_csv(FEATURES_CSV, index=False, encoding="utf-8-sig")


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    # Ensure unbuffered output for progress reporting
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

    options = vision.FaceLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=str(MODEL_PATH)),
        num_faces=1,
        min_face_detection_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # Process subject by subject — avoids massive sorted() on all 821K paths
    subject_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("T")],
                          key=lambda p: int(p.name.lstrip("T")))
    total_subjects = len(subject_dirs)
    print(f"Found {total_subjects} subject folders in {INPUT_DIR.name}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print(flush=True)

    if FEATURES_CSV.exists():
        FEATURES_CSV.unlink()

    rows = []
    failed = []
    processed = 0
    detected = 0
    is_first_chunk = True
    t0 = datetime.now()

    for subj_dir in subject_dirs:
        subj_name = subj_dir.name
        subj_images = sorted(subj_dir.glob("*.jpg"))
        n_subj = len(subj_images)

        for img_path in subj_images:
            try:
                feats = process_image(img_path, landmarker)
                if feats:
                    rows.append(feats)
                    detected += 1
                else:
                    parts = img_path.stem.split("_")
                    failed.append({
                        "image_path": str(img_path),
                        "image_name": img_path.name,
                        "subject_id": parts[0] if parts else "",
                        "task_id": parts[-1] if len(parts) > 1 else "",
                        "reason": "no_face_detected",
                    })
            except Exception as exc:
                parts = img_path.stem.split("_")
                failed.append({
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "subject_id": parts[0] if parts else "",
                    "task_id": parts[-1] if len(parts) > 1 else "",
                    "reason": str(exc)[:200],
                })

            processed += 1

            if len(rows) >= CHUNK_SIZE:
                save_chunk(rows, is_first_chunk)
                is_first_chunk = False
                rows.clear()

            if processed % 5000 == 0:
                elapsed = (datetime.now() - t0).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  [{processed} done] {detected} faces | "
                      f"{rate:.1f} img/s | failed={len(failed)}",
                      flush=True)

        # Per-subject progress
        elapsed = (datetime.now() - t0).total_seconds()
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  [{subj_name}] {n_subj} imgs | total={processed} | "
              f"{detected} faces | {rate:.1f} img/s", flush=True)

    # Final flush
    if rows:
        save_chunk(rows, is_first_chunk)
        rows.clear()

    landmarker.close()
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nDone in {elapsed/3600:.2f}h ({processed/elapsed:.1f} img/s)", flush=True)
    print(f"Faces: {detected}/{processed} ({100*detected/processed:.1f}%)", flush=True)
    print(f"Failed: {len(failed)}", flush=True)

    # --- Failed log ---
    if failed:
        pd.DataFrame(failed).to_csv(FAILED_LOG, index=False, encoding="utf-8-sig")
        print(f"Failed log: {FAILED_LOG} ({len(failed)} rows)", flush=True)

    # --- Read back combined features for summary ---
    print("Loading combined features for summary...")
    df = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    print(f"Features: {len(df)} rows x {len(df.columns)} cols", flush=True)

    # --- Summary per subject per condition ---
    skip_cols = {"image_path", "image_name", "subject_id", "task_id", "condition", "face_detected"}
    feat_cols = [c for c in df.columns if c not in skip_cols]

    summaries = []
    for (subj, task), g in df.groupby(["subject_id", "task_id"]):
        row = {"subject_id": subj, "task_id": task,
               "condition": g["condition"].iloc[0], "image_count": len(g)}
        for col in feat_cols:
            vals = g[col].dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) > 0 else np.nan
            row[f"{col}_std"] = vals.std() if len(vals) > 1 else np.nan
            row[f"{col}_median"] = vals.median() if len(vals) > 0 else np.nan
            row[f"{col}_min"] = vals.min() if len(vals) > 0 else np.nan
            row[f"{col}_max"] = vals.max() if len(vals) > 0 else np.nan
        summaries.append(row)

    df_s = pd.DataFrame(summaries)
    df_s.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"Summary CSV: {SUMMARY_CSV} ({len(df_s)} rows)", flush=True)

    # --- Stats ---
    print(f"\n--- Summary ---")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Conditions: {sorted(df['task_id'].unique())}")
    print(f"Images per condition:")
    for tid, cnt in df["task_id"].value_counts().items():
        print(f"  {tid}: {cnt}")

    print(f"\nKey features (mean across all images):")
    for k in ["ear_avg", "yaw_deg", "pitch_deg", "roll_deg",
              "au12_lip_corner_pull", "au25_lips_part", "au_smile_index"]:
        if k in df.columns:
            v = df[k].dropna()
            if len(v) > 0:
                print(f"  {k}: mean={v.mean():.4f}  std={v.std():.4f}  range=[{v.min():.4f}, {v.max():.4f}]")

    print(f"\nOutput folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
