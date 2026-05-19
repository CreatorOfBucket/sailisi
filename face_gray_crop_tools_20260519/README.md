# Face grayscale crop tools

This folder contains the script used to crop grayscale face-region images from the `T001` to `T158` subject folders.

## Scope

- Input layout: `<root>/T001/Image`, `<root>/T002/Image`, ...
- Processed files: image names whose stem ends with `T12`, `T22`, or `T32`.
- Output: grayscale cropped JPG files, grouped by subject folder.
- Extra outputs: per-range summary CSVs, crop-box CSVs, and failed-image CSVs.

The crop is intentionally conservative: it prioritizes keeping the complete visible face in frame over tightly cropping only the face. This is useful for downstream face3.0, AU, and EAR feature extraction.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Example

```powershell
python crop_face_grayscale.py `
  --root "D:\信任数据集\imag-gsr-ecg" `
  --output "C:\Users\26354\Desktop\saili数据处理\face_gray_T001_T158_T12_T22_T32_20260519_complete_visible" `
  --start 1 `
  --end 158 `
  --workers 8
```

For large datasets, running in smaller ranges is easier to monitor:

```powershell
python crop_face_grayscale.py --root "D:\信任数据集\imag-gsr-ecg" --output "<output_dir>" --start 1 --end 40 --workers 8
python crop_face_grayscale.py --root "D:\信任数据集\imag-gsr-ecg" --output "<output_dir>" --start 41 --end 80 --workers 8
python crop_face_grayscale.py --root "D:\信任数据集\imag-gsr-ecg" --output "<output_dir>" --start 81 --end 120 --workers 8
python crop_face_grayscale.py --root "D:\信任数据集\imag-gsr-ecg" --output "<output_dir>" --start 121 --end 158 --workers 8
```
