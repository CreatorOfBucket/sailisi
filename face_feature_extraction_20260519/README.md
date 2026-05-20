# 人脸特征提取脚本说明

本目录包含用于从灰度人脸图像中提取人脸相关特征的脚本，当前版本使用 MediaPipe Tasks API 的 `FaceLandmarker` 模型。

## 脚本列表

| 脚本 | 用途 | 输出 |
| --- | --- | --- |
| `extract_t001_features.py` | 单独提取 `T001` 的 EAR、Face3D、AU 特征 | `T001_features/EAR/`、`T001_features/Face3.0/`、`T001_features/Au/` |
| `extract_all_subjects_features.py` | 批量提取 `T002` 到 `T158` 的 EAR、Face3D、AU 特征，每个受试者单独输出 | `<subject>_features/EAR/`、`<subject>_features/Face3.0/`、`<subject>_features/Au/` |
| `extract_face_features.py` | 原始全量提取脚本，输出合并 CSV 和汇总 CSV，包含 HeadPose | `face_features_output/` |

## 当前默认路径

脚本默认读取以下输入目录：

```text
C:\Users\26354\Desktop\saili数据处理\face_gray_T001_T158_T12_T22_T32_20260519_complete_visible
```

MediaPipe 模型默认路径：

```text
C:\Users\26354\face_landmarker_v2.task
```

说明：`face_landmarker_v2.task` 使用 ASCII 路径是为了避免 MediaPipe 底层在中文路径下无法打开模型文件。

## 运行环境

已验证可用环境：

```text
Python:    C:\Anaconda\python.exe  (3.13.5)
numpy:     2.1.3
pandas:    2.2.3
mediapipe: 0.10.35
opencv:    4.13.0
```

其中 `mediapipe` 和 `opencv` 安装在用户级 site-packages。运行前建议显式设置：

```powershell
$env:PYTHONPATH='C:\Users\26354\AppData\Roaming\Python\Python313\site-packages'
```

## 使用方法

### 1. 提取 T001

```powershell
$env:PYTHONPATH='C:\Users\26354\AppData\Roaming\Python\Python313\site-packages'
& 'C:\Anaconda\python.exe' 'extract_t001_features.py'
```

可用 `--limit` 做少量帧测试：

```powershell
& 'C:\Anaconda\python.exe' 'extract_t001_features.py' --limit 20
```

### 2. 批量提取 T002-T158

```powershell
$env:PYTHONPATH='C:\Users\26354\AppData\Roaming\Python\Python313\site-packages'
& 'C:\Anaconda\python.exe' 'extract_all_subjects_features.py'
```

该脚本会自动跳过 `T001`，并为每个受试者生成独立目录，例如：

```text
T002_features/
  EAR/T002_ear.csv
  Face3.0/T002_face3d.csv
  Au/T002_au.csv
  T002_failed.csv
```

### 3. 全量合并输出

```powershell
$env:PYTHONPATH='C:\Users\26354\AppData\Roaming\Python\Python313\site-packages'
& 'C:\Anaconda\python.exe' 'extract_face_features.py'
```

该脚本输出到 `face_features_output/`，适合需要单一大 CSV 和汇总表的场景。

## 输出字段概览

`EAR` 输出包括：

- `image_name`
- `subject_id`
- `task_id`
- `condition`
- `ear_right`
- `ear_left`
- `ear_avg`

`Face3.0` 输出包括脸部宽高比例、鼻部深度、嘴部宽高、眼部尺寸、眼间距等三维/几何特征。

`Au` 输出包括基于关键点距离近似计算的 AU 特征，如眉毛抬起、嘴角拉伸、眼部闭合等。

## 注意事项

- 本目录只包含人脸特征提取脚本，不包含提取结果 CSV、视频、模型文件或原始图像。
- 批量脚本会覆盖同名输出 CSV。
- `*_failed.csv` 记录未检测到人脸或读取失败的图像；即使没有失败帧，也会保留表头，便于后续校验。
- 如果迁移到其他机器，需要同步修改脚本顶部的 `BASE`、`INPUT_DIR` 和 `MODEL_PATH`。
