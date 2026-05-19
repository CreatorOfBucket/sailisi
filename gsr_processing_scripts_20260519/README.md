# GSR Processing Scripts

This folder contains the scripts used to generate and post-process `ICSlab_project_GSR_output`.

## Files

- `GSRpreprocess.py`: splits raw Saili-style GSR `.npy` files by trigger, resamples to 50 Hz, converts resistance to conductance, runs `neurokit2.eda_process`, saves processed `.mat` files, and writes the original NeuroKit plots.
- `process_saili_gsr_batch.py`: batch entry point for folders such as `T001/GSR/*.npy`; calls `GSRpreprocess.process_and_save_gsr_segments`.
- `extract_scr_features.py`: extracts SCR/SCL summary features from `processedGSR/**/*.mat` into `GSR_SCR_features.csv`.
- `optimize_gsr_visualization.py`: regenerates clearer NeuroKit-based GSR plots by keeping the same `eda_process` signals and moving dense SCR event markers into a separate event strip.

## Typical local paths

```text
Input root:
C:\Users\26354\Desktop\saili数据处理

Output root:
C:\Users\26354\Desktop\saili数据处理\ICSlab_project_GSR_output

Project dependency root:
C:\Users\26354\Downloads\ICSlab-Human-Perception-Multimodal-Model-main\ICSlab-Human-Perception-Multimodal-Model-main
```

## Example commands

Generate processed `.mat` files and original NeuroKit plots:

```bash
python Preprocessing/process_saili_gsr_batch.py --input-root "C:\Users\26354\Desktop\saili数据处理"
```

Extract SCR features:

```bash
python extract_scr_features.py --input-dir "ICSlab_project_GSR_output/processedGSR" --output "ICSlab_project_GSR_output/GSR_SCR_features.csv"
```

Regenerate clearer plots without changing the NeuroKit processing algorithm:

```bash
python optimize_gsr_visualization.py \
  --data-root "C:\Users\26354\Desktop\saili数据处理" \
  --plot-root "C:\Users\26354\Desktop\saili数据处理\ICSlab_project_GSR_output\GSR_plots" \
  --project-root "C:\Users\26354\Downloads\ICSlab-Human-Perception-Multimodal-Model-main\ICSlab-Human-Perception-Multimodal-Model-main" \
  --subject-start 1 \
  --subject-end 158
```
