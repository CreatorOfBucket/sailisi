# ECG/GSR 数据处理工具

这个文件夹包含两类脚本：

- `process_ecg_gsr_fixed_sampling.py`：按固定采样率重建 ECG/GSR 时间戳并提取 `T12/T22/T32` 信任标签数据。
- `analyze_gsr_anomalies.py`：GSR 异常检测与可视化脚本，来自本地目录 `C:\Users\26354\Desktop\saili数据处理\GSR_anomaly_analysis_20260509`。

## 处理规则

ECG:

- 原始 ECG 每行包含 `raw ecg1` 到 `raw ecg50`。
- 按 200Hz 展开，每个采样点间隔 5ms。
- 不再使用相邻两行原始时间戳平均分。

GSR:

- 按 50Hz 重建时间戳，每个采样点间隔 20ms。

输出:

- 只保留 `T12/T22/T32`。
- 输出 CSV 只有三列：`Timestamp,Data,Trust`。
- 默认 `Trust` 映射为 `低/中/高`。

## 运行示例

```bash
python process_ecg_gsr_fixed_sampling.py --root "D:\信任数据集\imag-gsr-ecg" --start 1 --end 158 --trust-label zh
```

如需保留原始信任标签代码：

```bash
python process_ecg_gsr_fixed_sampling.py --root "D:\信任数据集\imag-gsr-ecg" --start 1 --end 158 --trust-label code
```

## GSR 异常分析

`analyze_gsr_anomalies.py` 会读取已处理的 GSR 合并文件，生成：

- 每个受试者的 GSR 时序图。
- 总体分布图。
- 异常受试者排名。
- 异常点明细。
- HTML 索引页。

异常候选规则包括：

- `Data <= 0`
- `Data == 1`
- `Data >= 1000`
- 相邻点突变 `>= 500`
- 受试者内 MAD 鲁棒离群点

这些规则用于筛查和人工复核，不等同于自动删除规则。
