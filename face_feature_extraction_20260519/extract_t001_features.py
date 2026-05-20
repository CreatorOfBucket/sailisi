"""
Extract EAR, Face3D, AU features for T001 only (all conditions: T12/T22/T32).
Output: T001_features/EAR/, T001_features/Au/, T001_features/Face3.0/
Uses pixel coordinates (fixed from previous normalized-coord bug).
"""
from pathlib import Path
from datetime import datetime
import argparse
import sys
import traceback

import cv2
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe import Image as MpImage, ImageFormat

BASE = Path(r"C:\Users\26354\Desktop\saili数据处理")
INPUT_DIR = BASE / "face_gray_T001_T158_T12_T22_T32_20260519_complete_visible" / "T001"
OUT_DIR = BASE / "T001_features"
MODEL_PATH = Path("C:/Users/26354/face_landmarker_v2.task")

# Output subdirectories
EAR_DIR = OUT_DIR / "EAR"
AU_DIR = OUT_DIR / "Au"
FACE3D_DIR = OUT_DIR / "Face3.0"
for d in [EAR_DIR, AU_DIR, FACE3D_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# MediaPipe landmark indices
RIGHT_EYE_CORNER_INNER = 133
RIGHT_EYE_CORNER_OUTER = 33
RIGHT_EYE_UPPER1 = 159
RIGHT_EYE_UPPER2 = 160
RIGHT_EYE_LOWER1 = 145
RIGHT_EYE_LOWER2 = 144
LEFT_EYE_CORNER_INNER = 362
LEFT_EYE_CORNER_OUTER = 263
LEFT_EYE_UPPER1 = 386
LEFT_EYE_UPPER2 = 385
LEFT_EYE_LOWER1 = 374
LEFT_EYE_LOWER2 = 373

NOSE_TIP = 4; CHIN = 152; FOREHEAD = 10
FACE_LEFT = 234; FACE_RIGHT = 454
MOUTH_LEFT = 61; MOUTH_RIGHT = 291
MOUTH_UPPER = 13; MOUTH_LOWER = 14
JAW_LEFT = 172; JAW_RIGHT = 397
NOSE_BOTTOM_L = 94; NOSE_BOTTOM_R = 334; NOSE_BASE = 5
RIGHT_EYEBROW_INNER = 107; LEFT_EYEBROW_INNER = 336
RIGHT_EYEBROW_OUTER = 55; LEFT_EYEBROW_OUTER = 285
RIGHT_EYEBROW_MID = 66; LEFT_EYEBROW_MID = 296
RIGHT_EYE_INDICES = [33, 133, 157, 158, 159, 160, 161, 163, 144, 145, 7, 173]
LEFT_EYE_INDICES = [362, 263, 384, 385, 386, 387, 388, 398, 374, 373, 381, 390]

# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def imread_unicode(path):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def _get_pt_2d(landmarks, idx, img_w, img_h):
    lm = landmarks[idx]
    return np.array([lm.x * img_w, lm.y * img_h])

# ---------------------------------------------------------------------------
# EAR (fixed: pixel coordinates)
# ---------------------------------------------------------------------------

def compute_ear(landmarks, img_w, img_h):
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
# Face3D (fixed: pixel coordinates for 2D, raw z for 3D)
# ---------------------------------------------------------------------------

def compute_face3d_features(landmarks, img_w, img_h):
    def _pt3(idx):
        p2 = _get_pt_2d(landmarks, idx, img_w, img_h)
        return np.array([p2[0], p2[1], landmarks[idx].z])

    left_eye_center = np.mean([_pt3(i) for i in (LEFT_EYE_CORNER_INNER, LEFT_EYE_CORNER_OUTER)], axis=0)
    right_eye_center = np.mean([_pt3(i) for i in (RIGHT_EYE_CORNER_INNER, RIGHT_EYE_CORNER_OUTER)], axis=0)
    ied = np.linalg.norm(left_eye_center[:2] - right_eye_center[:2])
    if ied < 1e-9:
        ied = 1.0

    nose = _pt3(NOSE_TIP); chin = _pt3(CHIN); forehead = _pt3(FOREHEAD)
    fl = _pt3(FACE_LEFT); fr = _pt3(FACE_RIGHT)
    m_lt = _pt3(MOUTH_LEFT); m_rt = _pt3(MOUTH_RIGHT)
    m_up = _pt3(MOUTH_UPPER); m_lo = _pt3(MOUTH_LOWER)
    j_lt = _pt3(JAW_LEFT); j_rt = _pt3(JAW_RIGHT)
    n_lt = _pt3(NOSE_BOTTOM_L); n_rt = _pt3(NOSE_BOTTOM_R)

    face_plane_z = (forehead[2] + chin[2]) / 2.0
    nose_protrusion = nose[2] - face_plane_z

    def _eye_size(indices):
        pts = np.array([_pt3(i) for i in indices])
        return ((np.max(pts[:, 1]) - np.min(pts[:, 1])) / ied,
                (np.max(pts[:, 0]) - np.min(pts[:, 0])) / ied)

    r_eh, r_ew = _eye_size(RIGHT_EYE_INDICES)
    l_eh, l_ew = _eye_size(LEFT_EYE_INDICES)

    def _dist2d(a, b):
        return np.linalg.norm(a[:2] - b[:2])

    return {
        "face3d_width_ratio": float(_dist2d(fr, fl) / ied),
        "face3d_height_ratio": float(_dist2d(forehead, chin) / ied),
        "face3d_nose_depth": float(nose[2]),
        "face3d_nose_protrusion": float(nose_protrusion),
        "face3d_nose_to_chin_ratio": float(_dist2d(nose, chin) / ied),
        "face3d_nose_to_forehead_ratio": float(_dist2d(nose, forehead) / ied),
        "face3d_mouth_width_ratio": float(_dist2d(m_rt, m_lt) / ied),
        "face3d_mouth_height_ratio": float(_dist2d(m_up, m_lo) / ied),
        "face3d_jaw_width_ratio": float(_dist2d(j_rt, j_lt) / ied),
        "face3d_nose_width_ratio": float(_dist2d(n_rt, n_lt) / ied),
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
# AU (fixed: pixel coordinates)
# ---------------------------------------------------------------------------

def compute_au_features(landmarks, img_w, img_h):
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

    r_eye_h = _dist2d(RIGHT_EYE_UPPER1, RIGHT_EYE_LOWER1) / norm
    l_eye_h = _dist2d(LEFT_EYE_UPPER1, LEFT_EYE_LOWER1) / norm
    r_eye_w = _dist2d(RIGHT_EYE_CORNER_INNER, RIGHT_EYE_CORNER_OUTER) / norm
    l_eye_w = _dist2d(LEFT_EYE_CORNER_INNER, LEFT_EYE_CORNER_OUTER) / norm

    mouth_w = _dist2d(MOUTH_LEFT, MOUTH_RIGHT) / norm

    j_lt = _get_pt_2d(landmarks, JAW_LEFT, img_w, img_h)
    j_rt = _get_pt_2d(landmarks, JAW_RIGHT, img_w, img_h)
    jaw_w = np.linalg.norm(j_rt - j_lt)

    return {
        "au1_inner_brow_raise_r": float(_dist2d(RIGHT_EYEBROW_INNER, RIGHT_EYE_CORNER_INNER) / norm),
        "au1_inner_brow_raise_l": float(_dist2d(LEFT_EYEBROW_INNER, LEFT_EYE_CORNER_INNER) / norm),
        "au2_outer_brow_raise_r": float(_dist2d(RIGHT_EYEBROW_OUTER, RIGHT_EYE_CORNER_OUTER) / norm),
        "au2_outer_brow_raise_l": float(_dist2d(LEFT_EYEBROW_OUTER, LEFT_EYE_CORNER_OUTER) / norm),
        "au4_brow_lower_r": float(_dist2d_pt(re_c, RIGHT_EYEBROW_MID) / norm),
        "au4_brow_lower_l": float(_dist2d_pt(le_c, LEFT_EYEBROW_MID) / norm),
        "au6_cheek_raise_r": float(r_eye_h),
        "au6_cheek_raise_l": float(l_eye_h),
        "au7_lid_tighten_r": float(r_eye_h),
        "au7_lid_tighten_l": float(l_eye_h),
        "au9_nose_wrinkle": float(_dist2d(NOSE_BOTTOM_L, NOSE_BOTTOM_R) / norm),
        "au10_upper_lip_raise": float(np.linalg.norm(mouth_up_pt - nose_base_pt) / norm),
        "au12_lip_corner_pull": float(mouth_w),
        "au15_lip_corner_depress_r": float((mouth_lt_pt[1] - mouth_c[1]) / norm),
        "au15_lip_corner_depress_l": float((mouth_rt_pt[1] - mouth_c[1]) / norm),
        "au17_chin_raise": float(np.linalg.norm(mouth_lo_pt - chin_pt) / norm),
        "au20_lip_stretch": float(mouth_w * norm / jaw_w if jaw_w > 0 else mouth_w),
        "au25_lips_part": float(_dist2d(MOUTH_UPPER, MOUTH_LOWER) / norm),
        "au26_jaw_drop": float(np.linalg.norm(mouth_lo_pt - chin_pt) / norm),
        "au43_eye_closure_r": float(r_eye_h * r_eye_w),
        "au43_eye_closure_l": float(l_eye_h * l_eye_w),
        "au_smile_index": float(mouth_w),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

    parser = argparse.ArgumentParser(description="Extract T001 EAR, Face3D, and AU features.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N frames for smoke testing.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    images = sorted(INPUT_DIR.glob("*.jpg"))
    total_available = len(images)
    if args.limit is not None:
        images = images[:max(args.limit, 0)]
    print(f"Found {len(images)} images for T001 in {INPUT_DIR}")
    if args.limit is not None:
        print(f"Limit: {len(images)}/{total_available} images")
    print(f"Output: {OUT_DIR}")
    print(f"  EAR    -> {EAR_DIR}")
    print(f"  Au     -> {AU_DIR}")
    print(f"  Face3.0 -> {FACE3D_DIR}")
    print()

    options = vision.FaceLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=str(MODEL_PATH)),
        num_faces=1, min_face_detection_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    ear_rows, au_rows, face3d_rows = [], [], []
    failed = []
    detected = 0
    t0 = datetime.now()

    for i, img_path in enumerate(images):
        try:
            img_bgr = imread_unicode(str(img_path))
            if img_bgr is None:
                failed.append({"image_name": img_path.name, "reason": "imread_failed"})
                continue

            ih, iw = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = MpImage(image_format=ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                failed.append({"image_name": img_path.name, "reason": "no_face"})
                continue

            landmarks = result.face_landmarks[0]
            detected += 1

            # Extract features
            ear = compute_ear(landmarks, iw, ih)
            f3d = compute_face3d_features(landmarks, iw, ih)
            au = compute_au_features(landmarks, iw, ih)

            # Parse task_id from filename: T001_YYYY-MM-DD_HH-MM-SS_XXX_Txx.jpg
            parts = img_path.stem.split("_")
            task_id = parts[-1] if parts else ""

            base = {
                "image_name": img_path.name,
                "subject_id": "T001",
                "task_id": task_id,
                "condition": {"T12": "low_trust", "T22": "medium_trust",
                              "T32": "high_trust"}.get(task_id, "unknown"),
            }

            ear_rows.append({**base, **ear})
            face3d_rows.append({**base, **f3d})
            au_rows.append({**base, **au})

        except Exception as exc:
            failed.append({"image_name": img_path.name, "reason": str(exc)[:200]})

        if (i + 1) % 1000 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(images)}] {detected} faces | {rate:.1f} img/s", flush=True)

    landmarker.close()
    elapsed = (datetime.now() - t0).total_seconds()

    # Save to separate CSVs
    if ear_rows:
        df_ear = pd.DataFrame(ear_rows)
        df_ear.to_csv(EAR_DIR / "T001_ear.csv", index=False, encoding="utf-8-sig")
        print(f"\nEAR: {len(df_ear)} rows -> {EAR_DIR / 'T001_ear.csv'}")

    if face3d_rows:
        df_f3d = pd.DataFrame(face3d_rows)
        df_f3d.to_csv(FACE3D_DIR / "T001_face3d.csv", index=False, encoding="utf-8-sig")
        print(f"Face3D: {len(df_f3d)} rows -> {FACE3D_DIR / 'T001_face3d.csv'}")

    if au_rows:
        df_au = pd.DataFrame(au_rows)
        df_au.to_csv(AU_DIR / "T001_au.csv", index=False, encoding="utf-8-sig")
        print(f"Au: {len(df_au)} rows -> {AU_DIR / 'T001_au.csv'}")

    failed_df = pd.DataFrame(failed, columns=["image_name", "reason"])
    failed_df.to_csv(OUT_DIR / "T001_failed.csv", index=False, encoding="utf-8-sig")
    print(f"Failed: {len(failed_df)} -> {OUT_DIR / 'T001_failed.csv'}")

    print(f"\nDone in {elapsed:.1f}s")
    det_rate = 100 * detected / len(images) if images else 0
    print(f"Faces: {detected}/{len(images)} ({det_rate:.1f}%)")


if __name__ == "__main__":
    main()
