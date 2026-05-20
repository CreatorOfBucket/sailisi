# ECG Processing Scripts

This folder contains the scripts used to generate and assess
`ICSlab_project_ECG_output` from Saili-style ECG/GSR subject folders.

## Files

- `reprocess_all.py`: re-cuts raw subject ECG/GSR CSV files by `T12/T22/T32`, channel-averages ECG, resamples ECG to 200 Hz and GSR to 50 Hz, and writes per-subject `processed_signal_data`.
- `extract_ecg_features.py`: reads `T*/processed_signal_data/ECG/*.csv`, extracts ECG/HRV-style signal features, writes `ICSlab_project_ECG_output/ECG_HRV_features.csv`, and generates `ECG_plots`.
- `analyze_ecg_anomalies.py`: reads processed ECG files, generates per-subject ECG anomaly plots, overview plots, CSV reports, and an HTML index under `ICSlab_project_ECG_output/ECG_anomaly_analysis`.
- `analyze_ecg_training_quality.py`: combines ECG feature and anomaly reports to generate keep/review/drop training-data recommendations under `ICSlab_project_ECG_output/ECG_training_quality_report`.

## Default Local Paths

These scripts were copied from the local Saili processing workspace and default to:

```text
Base data root:
C:\Users\26354\Desktop\saili_data

Output root:
C:\Users\26354\Desktop\saili_data\ICSlab_project_ECG_output
```

Expected source layout:

```text
C:\Users\26354\Desktop\saili_data\T001\ECG\*.csv
C:\Users\26354\Desktop\saili_data\T001\GSR\*.csv
C:\Users\26354\Desktop\saili_data\T002\ECG\*.csv
...
```

## Processing Order

Run from this folder or from the Saili data workspace.

1. Re-cut raw signals into per-subject processed CSV files:

```bash
python reprocess_all.py
```

This writes:

```text
Txxx/processed_signal_data/ECG/Txxx_ECG_T12_low_trust.csv
Txxx/processed_signal_data/ECG/Txxx_ECG_T22_medium_trust.csv
Txxx/processed_signal_data/ECG/Txxx_ECG_T32_high_trust.csv
```

2. Extract ECG features and generate per-condition ECG plots:

```bash
python extract_ecg_features.py ^
  --input-dir "C:\Users\26354\Desktop\saili_data" ^
  --output-dir "C:\Users\26354\Desktop\saili_data\ICSlab_project_ECG_output"
```

This writes:

```text
ICSlab_project_ECG_output/ECG_HRV_features.csv
ICSlab_project_ECG_output/ECG_plots/Txxx/*.png
```

3. Generate ECG anomaly analysis:

```bash
python analyze_ecg_anomalies.py
```

This writes:

```text
ICSlab_project_ECG_output/ECG_anomaly_analysis/index.html
ICSlab_project_ECG_output/ECG_anomaly_analysis/reports/ecg_anomaly_summary.csv
ICSlab_project_ECG_output/ECG_anomaly_analysis/reports/ecg_anomaly_points.csv
ICSlab_project_ECG_output/ECG_anomaly_analysis/per_subject_plots/*.png
ICSlab_project_ECG_output/ECG_anomaly_analysis/overview/*.png
```

4. Generate ECG training quality recommendations:

```bash
python analyze_ecg_training_quality.py
```

This writes:

```text
ICSlab_project_ECG_output/ECG_training_quality_report/ecg_subject_task_quality_detail.csv
ICSlab_project_ECG_output/ECG_training_quality_report/ecg_discard_recommendations.csv
ICSlab_project_ECG_output/ECG_training_quality_report/ecg_review_recommendations.csv
ICSlab_project_ECG_output/ECG_training_quality_report/ecg_subject_level_summary.csv
ICSlab_project_ECG_output/ECG_training_quality_report/ECG_training_data_quality_report.md
```

## Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:

- numpy
- pandas
- scipy
- matplotlib

## Notes

- The current ECG scripts are for the trust-label pipeline (`T12/T22/T32`), matching the existing `ICSlab_project_ECG_output` directory.
- `reprocess_all.py`, `analyze_ecg_anomalies.py`, and `analyze_ecg_training_quality.py` currently use the default local `BASE_DIR` in the script. Edit `BASE_DIR` if processing a different root.
- `extract_ecg_features.py` supports `--input-dir` and `--output-dir`.
