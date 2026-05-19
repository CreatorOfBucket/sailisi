"""
1. 从原始GSR数据中根据Trigger获取数据段。
2. 对每个数据段进行独立的预处理（电阻转电导、NeuroKit2清洗）。
3. 将每个预处理后的数据段保存为.mat文件。
"""

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.io import savemat
import warnings

# 忽略 NeuroKit2 中可能出现的 RuntimeWarning，例如当信号很短时
warnings.filterwarnings("ignore", category=RuntimeWarning)


def process_and_save_gsr_segments(gsr_data, subject_id, output_root):
    """
    加载单个GSR .npy文件，分割、预处理每个trigger段，并分别保存为.mat文件。

    Args:
        gsr_data (np.ndarray): 包含GSR数据的numpy数组 (可能已拼接)。
        subject_id (str): 当前被试的ID。
        output_root (str): 保存结果的根目录。
    """
    try:
        data = gsr_data

        if isinstance(data[0, 0], str): 
            columns = data[0]
            data = data[1:]
        else:
            columns = ["Timestamp", "Resistance(Koum)", "Trigger"]
            print(f"警告: 受试者 {subject_id} 的GSR数据未检测到表头，将使用默认列名。")

        df_raw = pd.DataFrame(data, columns=columns)


        df_raw['parsed_time'] = pd.to_datetime(
            df_raw['Timestamp'].astype(str).apply(lambda x: x + '000' if '.' in x and len(x.split('.')[-1]) < 6 else x),
            format="%y-%m-%d %H:%M:%S.%f",
            errors='coerce'
        )

        df_raw['Resistance(Koum)'] = pd.to_numeric(df_raw['Resistance(Koum)'], errors='coerce')
        df_raw.dropna(subset=['parsed_time', 'Resistance(Koum)'], inplace=True)

        if df_raw.empty:
            print(f"跳过受试者 {subject_id}: 时间戳或电阻数据无效。")
            return

        df_raw = df_raw.sort_values("parsed_time").reset_index(drop=True)

        time_series = df_raw["parsed_time"]
        dt = time_series.diff().dt.total_seconds().median()
        actual_fs = 1.0 / dt if dt and dt > 0 else np.nan
        print(f"受试者 {subject_id}: 实际采样率约为 {actual_fs:.2f} Hz，准备上采样到 50Hz")

        target_fs = 50.0
        target_interval_sec = 1.0 / target_fs
        new_time = pd.date_range(
            start=time_series.iloc[0],
            end=time_series.iloc[-1],
            freq=pd.to_timedelta(target_interval_sec, unit="s")
        )

        # 电阻：线性插值
        original_seconds = (time_series - time_series.iloc[0]).dt.total_seconds().values
        new_seconds = (new_time - time_series.iloc[0]).total_seconds()
        interp_resistance = np.interp(
            new_seconds,
            original_seconds,
            df_raw["Resistance(Koum)"].astype(float).values
        )

        trigger_idx = np.searchsorted(original_seconds, new_seconds, side="left")
        trigger_idx = np.clip(trigger_idx, 0, len(df_raw) - 1)
        interp_trigger = df_raw["Trigger"].iloc[trigger_idx].values


        df_resampled = pd.DataFrame({
            "Timestamp": new_time.strftime("%y-%m-%d %H:%M:%S.%f").str[:-3],
            "Resistance(Koum)": interp_resistance,
            "Trigger": interp_trigger,
            "parsed_time": new_time
        })

        print(f"上采样完成：{len(df_raw)} → {len(df_resampled)} 样本 (10Hz)")
        df_full = df_resampled.copy()

        # === 全局采样率===
        timestamps_ns = df_full['parsed_time'].astype("datetime64[ns]").astype(np.int64)
        time_diffs = np.diff(timestamps_ns / 1e9)
        valid_diffs = time_diffs[time_diffs > 0]
        if len(valid_diffs) == 0:
            print(f"跳过受试者 {subject_id}: 无法计算有效时间间隔。")
            return

        median_diff = np.median(valid_diffs)
        if median_diff <= 0 or np.isnan(median_diff):
            print(f"跳过受试者 {subject_id}: 采样率计算失败。")
            return

        sampling_rate = int(round(1 / median_diff))
        if sampling_rate < 2:
            print(f"跳过受试者 {subject_id}: 采样率过低 ({sampling_rate} Hz)。")
            return
        print(f"受试者 {subject_id} 的GSR数据全局采样率为: {sampling_rate} Hz")

        # === 清洗并识别有效 Trigger 段 ===
        trigger_corrections = {
            "anxitey": "anxiety",
            "terrify": "disgust"
        }

        df_full["Trigger"] = df_full["Trigger"].astype(str).str.lower().str.strip()
        df_full["Trigger"] = df_full["Trigger"].replace(trigger_corrections)

        invalid_triggers = {'none', 'no trigger', 'nano', 'nonenone', 'base.', 'hpaay', 'none.'}
        df_valid = df_full[
            df_full['Trigger'].notna() &
            (df_full['Trigger'] != '') &
            (~df_full['Trigger'].isin(invalid_triggers))
        ].copy()

        if df_valid.empty:
            print(f"在受试者 {subject_id} 的数据中未找到有效的Trigger段。")
            return


        df_valid['trigger_group'] = (df_valid['Trigger'] != df_valid['Trigger'].shift()).cumsum()
        segments = df_valid.groupby(['trigger_group', 'Trigger']).agg(
            start_time=('parsed_time', 'first'),
            end_time=('parsed_time', 'last')
        ).reset_index()

        print(f"在受试者 {subject_id} 中找到 {len(segments)} 个有效数据段。")

        # === 预处理并保存 ===
        output_dir = os.path.join(output_root, "processedGSR", str(int(subject_id)))
        plot_output_dir = os.path.join(output_root, "GSR_plots", subject_id)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plot_output_dir, exist_ok=True)

        for idx, row in segments.iterrows():
            trigger_name = row['Trigger']
            start_time = row['start_time']
            end_time = row['end_time']

            segment_df = df_full[
                (df_full['parsed_time'] >= start_time) &
                (df_full['parsed_time'] <= end_time)
            ].copy()

            if len(segment_df) < max(2 * sampling_rate, 10):  # 至少2秒或10点
                print(f"  - 跳过Trigger '{trigger_name}': 数据过短（{len(segment_df)} 点）。")
                continue


            safe_trigger_name = "".join(c for c in trigger_name if c.isalnum() or c in ('_', '-')).rstrip()
            if not safe_trigger_name:
                safe_trigger_name = "unknown"

            # 电阻 → 电导 (µS)
            resistance_kohm = segment_df["Resistance(Koum)"].astype(float).values
            resistance_kohm[resistance_kohm == 0] = 1e-9
            conductance_us = 1000.0 / resistance_kohm

            # NeuroKit2 预处理
            try:
                signals, info = nk.eda_process(conductance_us, sampling_rate=sampling_rate)
                cleaned_conductance = signals["EDA_Clean"].values
            except Exception as e:
                print(f"  - NeuroKit2 预处理失败（Trigger '{trigger_name}'）: {e}")
                continue


            try:
                fig = nk.eda_plot(signals, info)
                plot_filename = f"{subject_id}_{safe_trigger_name}_plot.png"
                plot_save_path = os.path.join(plot_output_dir, plot_filename)
                plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  - 图表已保存: {plot_filename}")
            except Exception as plot_e:
                print(f"  - 绘图失败（Trigger '{trigger_name}'）: {plot_e}")


            timestamps_raw = segment_df["Timestamp"].astype(str).values
            min_len = min(len(timestamps_raw), len(cleaned_conductance))
            if min_len == 0:
                print(f"  - 跳过Trigger '{trigger_name}': 预处理后数据为空。")
                continue


            mat_content = {
                "data": cleaned_conductance[:min_len].reshape(-1, 1),
                "Timestamp": timestamps_raw[:min_len].reshape(-1, 1),
                "task_id": trigger_name,
                "user_id": str(int(subject_id))
            }

            output_mat_path = os.path.join(output_dir, f"{str(int(subject_id))}_{safe_trigger_name}.mat")
            savemat(output_mat_path, {"mat_dict": mat_content})
            print(f"  - 成功处理并保存: {os.path.basename(output_mat_path)}")

    except Exception as e:
        print(f"处理受试者 {subject_id} 的GSR数据时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
