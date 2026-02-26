#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機器人速度測試數據分析程式 v4

改進項目：
1. 嚴格穩態誤差 (SSE) 定義：採用控制理論標準，使用 ±5% 容許範圍判定穩態
2. 新增穩定時間 (Settling Time) 計算
3. 新增線速度/角速度追蹤性能專用圖表
4. 報告中詳細說明各項計算方法

使用方式: python3 speed_analysis_v4.py <csv_file>
輸出：圖表和報告儲存於 speed_analysis_v4_<filename>/ 目錄
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys
from datetime import datetime

# ============================================================
# 全域參數
# ============================================================

# 穩態誤差判定參數
TOLERANCE_PERCENT = 0.05        # 容許範圍 ±5%
MIN_STEADY_SAMPLES = 10         # 穩態確認最少連續樣本數

# 使用非互動式後端
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 資料載入
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:
    """載入並預處理數據"""
    df = pd.read_csv(filepath)
    # 轉為相對時間 (秒)
    df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    return df


# ============================================================
# 測試概覽與基本統計
# ============================================================

def analyze_test_overview(df: pd.DataFrame) -> dict:
    """測試概覽分析"""
    duration = df['time'].max()
    return {
        'total_samples': len(df),
        'duration_sec': round(duration, 1),
        'duration_min': round(duration / 60, 2),
        'sample_rate_hz': round(len(df) / duration, 1),
        'start_timestamp': df['timestamp'].iloc[0],
        'end_timestamp': df['timestamp'].iloc[-1],
    }


def analyze_speed_extremes(df: pd.DataFrame) -> dict:
    """速度極值分析"""
    return {
        # 指令速度範圍
        'cmd_lin_x_min': round(df['cmd_lin_x'].min(), 3),
        'cmd_lin_x_max': round(df['cmd_lin_x'].max(), 3),
        'cmd_ang_z_min': round(df['cmd_ang_z'].min(), 3),
        'cmd_ang_z_max': round(df['cmd_ang_z'].max(), 3),
        # 實際達到的最高速度
        'actual_lin_max': round(df['odom_lin_speed'].max(), 4),
        'actual_ang_max': round(max(abs(df['odom_ang_z'].min()), abs(df['odom_ang_z'].max())), 4),
        # 實際達到最高速度時的命令
        'cmd_at_max_lin': round(df.loc[df['odom_lin_speed'].idxmax(), 'cmd_lin_x'], 3),
        'cmd_at_max_ang': round(df.loc[df['odom_ang_z'].abs().idxmax(), 'cmd_ang_z'], 3),
    }


# ============================================================
# 嚴格穩態誤差分析 (核心改進)
# ============================================================

def find_settling_time(actual_values: np.ndarray, cmd_val: float, 
                       tolerance_percent: float = TOLERANCE_PERCENT,
                       min_steady_samples: int = MIN_STEADY_SAMPLES) -> dict:
    """
    計算穩定時間 (Settling Time)
    
    定義：
    - 系統輸出首次進入並「持續」停留在容許範圍內的時間點
    - 容許範圍 = cmd_val ± (tolerance_percent × |cmd_val|)
    
    回傳：
    - settling_idx: 穩定時間的索引位置，若無法達到穩態則為 None
    - is_settled: 是否達到穩態
    - final_value: 穩態區段平均值（若達到穩態）
    """
    if cmd_val == 0:
        # 對於 cmd=0 的情況，使用絕對容許範圍
        tolerance = 0.005  # 5 mm/s 絕對容許
    else:
        tolerance = abs(cmd_val * tolerance_percent)
    
    lower_bound = cmd_val - tolerance
    upper_bound = cmd_val + tolerance
    
    n = len(actual_values)
    
    # 滑動窗口檢查
    for i in range(n - min_steady_samples + 1):
        window = actual_values[i:i + min_steady_samples]
        # 檢查窗口內所有值是否都在容許範圍內
        if np.all((window >= lower_bound) & (window <= upper_bound)):
            # 找到穩態起始點
            steady_data = actual_values[i:]
            return {
                'settling_idx': i,
                'is_settled': True,
                'final_value': np.mean(steady_data),
                'tolerance': tolerance,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    # 無法達到穩態
    return {
        'settling_idx': None,
        'is_settled': False,
        'final_value': np.mean(actual_values[-min_steady_samples:]) if n >= min_steady_samples else np.mean(actual_values),
        'tolerance': tolerance,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def analyze_linear_tracking_strict(df: pd.DataFrame) -> pd.DataFrame:
    """
    線速度追蹤性能分析（嚴格穩態誤差版本）
    
    計算方法：
    1. 對每個命令值，取出該命令期間的所有資料
    2. 使用 ±5% 容許範圍判定穩態
    3. 計算穩定時間 (settling_time)
    4. 僅使用穩態後的資料計算 SSE
    """
    results = []
    cmd_values = sorted(df['cmd_lin_x'].unique())
    sample_rate = len(df) / df['time'].max()  # 估算取樣率
    
    for cmd_val in cmd_values:
        if cmd_val <= 0:
            continue
        
        subset = df[df['cmd_lin_x'] == cmd_val]
        if len(subset) < MIN_STEADY_SAMPLES:
            continue
        
        actual_values = subset['odom_lin_speed'].values
        times = subset['time'].values
        
        # 計算穩定時間
        settling_result = find_settling_time(actual_values, cmd_val)
        
        if settling_result['is_settled']:
            settling_idx = settling_result['settling_idx']
            # 取穩態後資料
            steady_data = actual_values[settling_idx:]
            steady_times = times[settling_idx:]
            settling_time_ms = (times[settling_idx] - times[0]) * 1000 if settling_idx > 0 else 0
        else:
            # 未達穩態，使用後 50% 資料作為近似
            midpoint = len(actual_values) // 2
            steady_data = actual_values[midpoint:]
            settling_time_ms = None
        
        actual_mean = np.mean(steady_data)
        actual_std = np.std(steady_data)
        
        # 穩態誤差計算
        sse_absolute = cmd_val - actual_mean
        sse_percent = (sse_absolute / cmd_val) * 100 if cmd_val > 0 else 0
        
        # RMSE 計算
        errors = cmd_val - steady_data
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # 追蹤率
        tracking_ratio = (actual_mean / cmd_val) * 100 if cmd_val > 0 else 0
        
        # 超調量 (Overshoot)
        max_value = np.max(actual_values)
        overshoot_percent = ((max_value - cmd_val) / cmd_val) * 100 if max_value > cmd_val else 0
        
        results.append({
            'cmd_lin_x': round(cmd_val, 2),
            'actual_mean': round(actual_mean, 4),
            'actual_std': round(actual_std, 4),
            'sse_absolute': round(sse_absolute, 4),
            'sse_percent': round(sse_percent, 2),
            'rmse': round(rmse, 4),
            'tracking_ratio_pct': round(tracking_ratio, 1),
            'settling_time_ms': round(settling_time_ms, 1) if settling_time_ms is not None else None,
            'is_settled': settling_result['is_settled'],
            'overshoot_pct': round(overshoot_percent, 2),
            'samples': len(steady_data),
            'total_samples': len(actual_values)
        })
    
    return pd.DataFrame(results)


def analyze_angular_tracking_strict(df: pd.DataFrame) -> pd.DataFrame:
    """
    角速度追蹤性能分析（嚴格穩態誤差版本）
    """
    results = []
    cmd_values = sorted(df['cmd_ang_z'].unique())
    
    for cmd_val in cmd_values:
        if abs(cmd_val) < 0.1:
            continue
        
        subset = df[df['cmd_ang_z'] == cmd_val]
        if len(subset) < MIN_STEADY_SAMPLES:
            continue
        
        actual_values = subset['odom_ang_z'].values
        times = subset['time'].values
        
        # 計算穩定時間
        settling_result = find_settling_time(actual_values, cmd_val)
        
        if settling_result['is_settled']:
            settling_idx = settling_result['settling_idx']
            steady_data = actual_values[settling_idx:]
            settling_time_ms = (times[settling_idx] - times[0]) * 1000 if settling_idx > 0 else 0
        else:
            midpoint = len(actual_values) // 2
            steady_data = actual_values[midpoint:]
            settling_time_ms = None
        
        actual_mean = np.mean(steady_data)
        actual_std = np.std(steady_data)
        
        # 穩態誤差計算
        sse_absolute = cmd_val - actual_mean
        sse_percent = (sse_absolute / cmd_val) * 100 if abs(cmd_val) > 0.01 else 0
        
        # RMSE
        errors = cmd_val - steady_data
        rmse = np.sqrt(np.mean(errors ** 2))
        
        # 追蹤率
        tracking_ratio = (actual_mean / cmd_val) * 100 if abs(cmd_val) > 0.01 else 0
        
        results.append({
            'cmd_ang_z': round(cmd_val, 2),
            'actual_mean': round(actual_mean, 4),
            'actual_std': round(actual_std, 4),
            'sse_absolute': round(sse_absolute, 4),
            'sse_percent': round(sse_percent, 2),
            'rmse': round(rmse, 4),
            'tracking_ratio_pct': round(tracking_ratio, 1),
            'settling_time_ms': round(settling_time_ms, 1) if settling_time_ms is not None else None,
            'is_settled': settling_result['is_settled'],
            'samples': len(steady_data),
            'total_samples': len(actual_values)
        })
    
    return pd.DataFrame(results)


# ============================================================
# 延遲分析
# ============================================================

def analyze_latency(df: pd.DataFrame) -> tuple:
    """延遲分析：計算每次命令變化到實際速度響應的延遲"""
    df = df.copy()
    df['cmd_lin_change'] = df['cmd_lin_x'].diff().abs() > 0.01
    df['cmd_ang_change'] = df['cmd_ang_z'].diff().abs() > 0.01
    df['cmd_change'] = df['cmd_lin_change'] | df['cmd_ang_change']
    
    latencies = []
    latency_timeseries = []
    
    change_indices = df[df['cmd_change']].index.tolist()
    
    for idx in change_indices:
        if idx < 2 or idx > len(df) - 20:
            continue
        
        cmd_time = df.loc[idx, 'timestamp']
        event_time = df.loc[idx, 'time']
        cmd_lin_before = df.loc[idx - 1, 'cmd_lin_x']
        cmd_lin_after = df.loc[idx, 'cmd_lin_x']
        cmd_ang_before = df.loc[idx - 1, 'cmd_ang_z']
        cmd_ang_after = df.loc[idx, 'cmd_ang_z']
        
        # 線速度增加的情況
        if cmd_lin_after > cmd_lin_before:
            speed_before = df.loc[idx - 1, 'odom_lin_speed']
            threshold = speed_before + 0.008
            
            for j in range(idx, min(idx + 30, len(df))):
                if df.loc[j, 'odom_lin_speed'] > threshold:
                    latency = (df.loc[j, 'timestamp'] - cmd_time) * 1000
                    if 10 < latency < 500:
                        latencies.append({
                            'type': 'linear_accel',
                            'latency_ms': round(latency, 1),
                            'event_time': round(event_time, 2),
                            'cmd_before': round(cmd_lin_before, 3),
                            'cmd_after': round(cmd_lin_after, 3),
                        })
                        latency_timeseries.append({
                            'time': event_time,
                            'latency_ms': round(latency, 1),
                            'type': 'linear'
                        })
                    break
        
        # 角速度變化
        if abs(cmd_ang_after) > abs(cmd_ang_before) and abs(cmd_ang_after - cmd_ang_before) > 0.05:
            ang_before = abs(df.loc[idx - 1, 'odom_ang_z'])
            threshold = ang_before + 0.02
            
            for j in range(idx, min(idx + 30, len(df))):
                if abs(df.loc[j, 'odom_ang_z']) > threshold:
                    latency = (df.loc[j, 'timestamp'] - cmd_time) * 1000
                    if 10 < latency < 500:
                        latencies.append({
                            'type': 'angular_accel',
                            'latency_ms': round(latency, 1),
                            'event_time': round(event_time, 2),
                            'cmd_before': round(cmd_ang_before, 3),
                            'cmd_after': round(cmd_ang_after, 3),
                        })
                        latency_timeseries.append({
                            'time': event_time,
                            'latency_ms': round(latency, 1),
                            'type': 'angular'
                        })
                    break
    
    # 統計
    if latencies:
        lat_values = [l['latency_ms'] for l in latencies]
        stats = {
            'mean_ms': round(np.mean(lat_values), 1),
            'min_ms': round(np.min(lat_values), 1),
            'max_ms': round(np.max(lat_values), 1),
            'std_ms': round(np.std(lat_values), 1),
            'median_ms': round(np.median(lat_values), 1),
            'count': len(latencies),
        }
    else:
        stats = {'mean_ms': 0, 'count': 0}
    
    return stats, latency_timeseries


# ============================================================
# 煞車分析
# ============================================================

def analyze_braking(df: pd.DataFrame) -> list:
    """煞車特性分析"""
    braking_events = []
    
    i = 1
    while i < len(df) - 1:
        prev_cmd = df.loc[i - 1, 'cmd_lin_x']
        curr_cmd = df.loc[i, 'cmd_lin_x']
        prev_speed = df.loc[i - 1, 'odom_lin_speed']
        
        if prev_cmd > 0.05 and curr_cmd == 0 and prev_speed > 0.05:
            start_idx = i
            start_speed = prev_speed
            start_time = df.loc[i, 'timestamp']
            start_x = df.loc[i, 'odom_x']
            start_y = df.loc[i, 'odom_y']
            
            for j in range(i, min(i + 150, len(df))):
                if df.loc[j, 'odom_lin_speed'] < 0.005:
                    stop_time = df.loc[j, 'timestamp']
                    stop_x = df.loc[j, 'odom_x']
                    stop_y = df.loc[j, 'odom_y']
                    
                    duration_sec = stop_time - start_time
                    distance = np.sqrt((stop_x - start_x)**2 + (stop_y - start_y)**2)
                    avg_decel = start_speed / duration_sec if duration_sec > 0 else 0
                    
                    if 0.01 < duration_sec < 10:
                        braking_events.append({
                            'start_speed_mps': round(start_speed, 4),
                            'stop_time_ms': round(duration_sec * 1000, 0),
                            'slide_distance_m': round(distance, 4),
                            'avg_decel_mps2': round(avg_decel, 4),
                            'time_at_event': df.loc[i, 'time']
                        })
                    i = j
                    break
            else:
                i += 1
        else:
            i += 1
    
    return braking_events


# ============================================================
# 視覺化：速度對比圖
# ============================================================

def plot_velocity_comparison(df: pd.DataFrame, output_path: Path):
    """繪製線速度和角速度的命令 vs 實際值線圖"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # 線速度對比
    ax1 = axes[0]
    ax1.plot(df['time'], df['cmd_lin_x'], 'b-', label='Command (cmd_lin_x)', 
             alpha=0.8, linewidth=1.2)
    ax1.plot(df['time'], df['odom_lin_speed'], 'r-', label='Actual (odom_lin_speed)', 
             alpha=0.7, linewidth=1.0)
    ax1.set_ylabel('Linear Velocity (m/s)', fontsize=12)
    ax1.set_title('Linear Velocity: Command vs Actual', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(df['time'], df['cmd_lin_x'], df['odom_lin_speed'], 
                     alpha=0.2, color='purple', label='Tracking Error')
    
    # 角速度對比
    ax2 = axes[1]
    ax2.plot(df['time'], df['cmd_ang_z'], 'b-', label='Command (cmd_ang_z)', 
             alpha=0.8, linewidth=1.2)
    ax2.plot(df['time'], df['odom_ang_z'], 'r-', label='Actual (odom_ang_z)', 
             alpha=0.7, linewidth=1.0)
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_title('Angular Velocity: Command vs Actual', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================
# 視覺化：線速度追蹤性能專用圖（新增）
# ============================================================

def plot_linear_tracking_performance(df: pd.DataFrame, lin_tracking: pd.DataFrame, output_path: Path):
    """
    繪製線速度追蹤性能專用圖
    
    樣式：
    - 左圖：Command vs Actual 散點圖（帶 Ideal 對角線）
    - 右圖：追蹤率條形圖
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === 左圖：Command vs Actual 散點圖 ===
    ax1 = axes[0]
    
    if not lin_tracking.empty:
        cmd_vals = lin_tracking['cmd_lin_x'].values
        actual_vals = lin_tracking['actual_mean'].values
        
        # 繪製 Ideal 對角線
        max_val = max(cmd_vals.max(), actual_vals.max()) * 1.1
        ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.7, label='Ideal')
        
        # 繪製實際數據點
        ax1.plot(cmd_vals, actual_vals, 'o-', color='steelblue', markersize=6, 
                 linewidth=1.5, alpha=0.8)
        
        ax1.set_xlim([0, max_val])
        ax1.set_ylim([0, max_val])
    
    ax1.set_xlabel('Command Velocity (m/s)', fontsize=12)
    ax1.set_ylabel('Actual Velocity (m/s)', fontsize=12)
    ax1.set_title('Linear Velocity Tracking', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # === 右圖：追蹤率條形圖 ===
    ax2 = axes[1]
    
    if not lin_tracking.empty:
        cmd_vals = lin_tracking['cmd_lin_x'].values
        tracking_ratios = lin_tracking['tracking_ratio_pct'].values
        
        # 繪製條形圖
        bars = ax2.bar(cmd_vals, tracking_ratios, width=0.04, alpha=0.8, 
                       color='steelblue', edgecolor='black', linewidth=0.5)
        
        # 100% 參考線
        ax2.axhline(y=100, color='black', linestyle='--', linewidth=1.5)
        
        ax2.set_xlabel('Command Velocity (m/s)', fontsize=12)
        ax2.set_ylabel('Tracking Ratio (%)', fontsize=12)
        ax2.set_title('Velocity Tracking Ratio', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 動態調整 Y 軸範圍
        min_ratio = min(tracking_ratios)
        ax2.set_ylim([max(0, min_ratio - 5), 105])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================
# 視覺化：角速度追蹤性能專用圖（新增）
# ============================================================

def plot_angular_tracking_performance(df: pd.DataFrame, ang_tracking: pd.DataFrame, output_path: Path):
    """
    繪製角速度追蹤性能專用圖
    
    樣式：
    - 左圖：Command vs Actual 散點圖（帶 Ideal 對角線）
    - 右圖：追蹤率條形圖
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === 左圖：Command vs Actual 散點圖 ===
    ax1 = axes[0]
    
    if not ang_tracking.empty:
        cmd_vals = ang_tracking['cmd_ang_z'].values
        actual_vals = ang_tracking['actual_mean'].values
        
        # 繪製 Ideal 對角線（正負方向都要）
        max_abs = max(abs(cmd_vals).max(), abs(actual_vals).max()) * 1.1
        ax1.plot([-max_abs, max_abs], [-max_abs, max_abs], 'k--', linewidth=1.5, alpha=0.7, label='Ideal')
        
        # 繪製實際數據點
        ax1.plot(cmd_vals, actual_vals, 'o-', color='steelblue', markersize=6, 
                 linewidth=1.5, alpha=0.8)
        
        ax1.set_xlim([-max_abs, max_abs])
        ax1.set_ylim([-max_abs, max_abs])
    
    ax1.set_xlabel('Command Angular Velocity (rad/s)', fontsize=12)
    ax1.set_ylabel('Actual Angular Velocity (rad/s)', fontsize=12)
    ax1.set_title('Angular Velocity Tracking', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # === 右圖：追蹤率條形圖 ===
    ax2 = axes[1]
    
    if not ang_tracking.empty:
        cmd_vals = ang_tracking['cmd_ang_z'].values
        tracking_ratios = ang_tracking['tracking_ratio_pct'].values
        
        # 繪製條形圖
        bars = ax2.bar(cmd_vals, tracking_ratios, width=0.15, alpha=0.8, 
                       color='steelblue', edgecolor='black', linewidth=0.5)
        
        # 100% 參考線
        ax2.axhline(y=100, color='black', linestyle='--', linewidth=1.5)
        
        ax2.set_xlabel('Command Angular Velocity (rad/s)', fontsize=12)
        ax2.set_ylabel('Tracking Ratio (%)', fontsize=12)
        ax2.set_title('Angular Velocity Tracking Ratio', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 動態調整 Y 軸範圍
        min_ratio = min(tracking_ratios)
        ax2.set_ylim([max(0, min_ratio - 5), 105])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================
# 視覺化：延遲分析
# ============================================================

def plot_latency_analysis(latency_data: list, latency_stats: dict, output_path: Path):
    """繪製延遲分析圖"""
    if not latency_data:
        print(f"  ⚠ No latency data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    lin_data = [d for d in latency_data if d['type'] == 'linear']
    ang_data = [d for d in latency_data if d['type'] == 'angular']
    all_latencies = [d['latency_ms'] for d in latency_data]
    
    # 線速度延遲時序
    ax1 = axes[0, 0]
    if lin_data:
        times = [d['time'] for d in lin_data]
        lats = [d['latency_ms'] for d in lin_data]
        ax1.plot(times, lats, 'bo-', markersize=5, alpha=0.7, linewidth=1)
        ax1.axhline(y=np.mean(lats), color='b', linestyle='--', alpha=0.5, 
                    label=f'Mean: {np.mean(lats):.1f} ms')
    ax1.set_ylabel('Latency (ms)', fontsize=11)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_title('Linear Velocity Latency', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 角速度延遲時序
    ax2 = axes[0, 1]
    if ang_data:
        times = [d['time'] for d in ang_data]
        lats = [d['latency_ms'] for d in ang_data]
        ax2.plot(times, lats, 'go-', markersize=5, alpha=0.7, linewidth=1)
        ax2.axhline(y=np.mean(lats), color='g', linestyle='--', alpha=0.5, 
                    label=f'Mean: {np.mean(lats):.1f} ms')
    ax2.set_ylabel('Latency (ms)', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_title('Angular Velocity Latency', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 直方圖
    ax3 = axes[1, 0]
    ax3.hist(all_latencies, bins=25, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(x=latency_stats.get('mean_ms', 0), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {latency_stats.get('mean_ms', 0)} ms")
    ax3.axvline(x=latency_stats.get('median_ms', 0), color='orange', linestyle='--', 
                linewidth=2, label=f"Median: {latency_stats.get('median_ms', 0)} ms")
    ax3.set_xlabel('Latency (ms)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Latency Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # CDF
    ax4 = axes[1, 1]
    sorted_lat = np.sort(all_latencies)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat) * 100
    ax4.plot(sorted_lat, cdf, 'b-', linewidth=2)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=95, color='gray', linestyle='--', alpha=0.5)
    p50 = np.percentile(all_latencies, 50)
    p95 = np.percentile(all_latencies, 95)
    ax4.scatter([p50, p95], [50, 95], c='red', s=50, zorder=5)
    ax4.annotate(f'{p50:.0f}ms', (p50, 50), xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.annotate(f'{p95:.0f}ms', (p95, 95), xytext=(5, -10), textcoords='offset points', fontsize=9)
    ax4.set_xlabel('Latency (ms)', fontsize=11)
    ax4.set_ylabel('Cumulative %', fontsize=11)
    ax4.set_title('Latency CDF', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================
# 視覺化：煞車分析
# ============================================================

def plot_braking_analysis(braking_events: list, output_path: Path):
    """繪製煞車分析圖"""
    if not braking_events:
        print(f"  ⚠ No braking events to plot")
        return
    
    speeds = [b['start_speed_mps'] for b in braking_events]
    distances = [b['slide_distance_m'] for b in braking_events]
    times = [b['stop_time_ms'] for b in braking_events]
    decels = [b['avg_decel_mps2'] for b in braking_events]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 初速 vs 滑行距離
    ax = axes[0, 0]
    ax.scatter(speeds, distances, alpha=0.7, s=40, c='blue', edgecolors='black', linewidths=0.3)
    if len(speeds) > 2:
        z = np.polyfit(speeds, distances, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(speeds), max(speeds), 100)
        ax.plot(x_fit, p(x_fit), 'r--', alpha=0.7, label=f'Fit: d = {z[0]:.3f}v + {z[1]:.3f}')
        ax.legend(fontsize=9)
    ax.set_xlabel('Initial Speed (m/s)', fontsize=11)
    ax.set_ylabel('Sliding Distance (m)', fontsize=11)
    ax.set_title('Speed vs Sliding Distance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 初速 vs 停止時間
    ax = axes[0, 1]
    ax.scatter(speeds, times, alpha=0.7, s=40, c='green', edgecolors='black', linewidths=0.3)
    ax.set_xlabel('Initial Speed (m/s)', fontsize=11)
    ax.set_ylabel('Stop Time (ms)', fontsize=11)
    ax.set_title('Speed vs Stop Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 減速度分佈
    ax = axes[1, 0]
    ax.hist(decels, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(x=np.mean(decels), color='red', linestyle='--', 
               label=f'Mean: {np.mean(decels):.3f} m/s²')
    ax.set_xlabel('Deceleration (m/s²)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Deceleration Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 減速度 CDF
    ax = axes[1, 1]
    sorted_decels = np.sort(decels)
    cdf = np.arange(1, len(sorted_decels) + 1) / len(sorted_decels) * 100
    ax.plot(sorted_decels, cdf, 'b-', linewidth=2)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Deceleration (m/s²)', fontsize=11)
    ax.set_ylabel('Cumulative %', fontsize=11)
    ax.set_title('Deceleration CDF', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ============================================================
# 報告生成
# ============================================================

def generate_report(df: pd.DataFrame, output_path: Path, 
                    overview: dict, extremes: dict, 
                    lin_tracking: pd.DataFrame, ang_tracking: pd.DataFrame,
                    latency_stats: dict, braking_events: list):
    """生成完整 Markdown 報告（含方法論說明）"""
    
    report = f"""# 機器人速度測試分析報告 v4

**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**數據來源**: {df.attrs.get('source', 'speed_test.csv')}

---

## 方法論說明

本報告採用控制理論中更嚴謹的穩態誤差 (Steady-State Error, SSE) 定義。

### 穩態判定條件

| 參數 | 數值 | 說明 |
|------|------|------|
| 容許範圍 (Tolerance) | ±{TOLERANCE_PERCENT*100:.0f}% | 相對於命令值的允許偏差 |
| 最小穩態樣本數 | {MIN_STEADY_SAMPLES} 個 | 連續在容許範圍內的最少樣本數 |

### 穩定時間 (Settling Time, ts) 計算方法

1. 對於命令值 `cmd_val`，計算容許範圍：
   - 下界 = `cmd_val × (1 - {TOLERANCE_PERCENT})`
   - 上界 = `cmd_val × (1 + {TOLERANCE_PERCENT})`
2. 遍歷實際速度時序資料
3. 找到「首次進入容許範圍且後續連續 {MIN_STEADY_SAMPLES} 個樣本都在範圍內」的時間點
4. 若整個命令期間都無法達到此條件，標記為「未達穩態」

### 穩態誤差 (SSE) 計算公式

```
SSE_absolute = cmd_val - actual_mean (穩態後)
SSE_percent = (SSE_absolute / cmd_val) × 100%
```

### 追蹤率計算公式

```
Tracking Ratio = (actual_mean / cmd_val) × 100%
```

> **註**：追蹤率 = 100% - SSE%，理想值為 100%

### RMSE (均方根誤差) 計算公式

```
RMSE = √(Σ(cmd_val - actual_i)² / n)
```

---

## 1. 測試概覽

| 項目 | 數值 |
|------|------|
| 總樣本數 | {overview['total_samples']} |
| 測試時長 | {overview['duration_sec']} 秒 ({overview['duration_min']} 分鐘) |
| 取樣率 | {overview['sample_rate_hz']} Hz |
| 開始時間戳 | {overview['start_timestamp']:.4f} |
| 結束時間戳 | {overview['end_timestamp']:.4f} |

---

## 2. 速度極值

### 指令速度範圍

| 速度類型 | 最小值 | 最大值 |
|----------|--------|--------|
| 線速度 (cmd_lin_x) | {extremes['cmd_lin_x_min']} m/s | {extremes['cmd_lin_x_max']} m/s |
| 角速度 (cmd_ang_z) | {extremes['cmd_ang_z_min']} rad/s | {extremes['cmd_ang_z_max']} rad/s |

### 實際達到的最高速度

| 速度類型 | 最高實際值 | 當時命令值 |
|----------|------------|------------|
| 線速度 | {extremes['actual_lin_max']} m/s | {extremes['cmd_at_max_lin']} m/s |
| 角速度 | {extremes['actual_ang_max']} rad/s | {extremes['cmd_at_max_ang']} rad/s |

---

## 3. 線速度追蹤性能（嚴格穩態誤差）

| 命令 (m/s) | 實際 (m/s) | Ts (ms) | SSE (m/s) | SSE (%) | RMSE | 追蹤率 (%) | 穩態 | 超調 (%) |
|------------|------------|---------|-----------|---------|------|------------|------|----------|
"""
    
    for _, row in lin_tracking.iterrows():
        ts_str = f"{row['settling_time_ms']:.0f}" if row['settling_time_ms'] is not None else "N/A"
        settled_str = "✓" if row['is_settled'] else "✗"
        overshoot = row.get('overshoot_pct', 0)
        report += f"| {row['cmd_lin_x']:.2f} | {row['actual_mean']:.4f} | {ts_str} | {row['sse_absolute']:.4f} | {row['sse_percent']:.1f}% | {row['rmse']:.4f} | {row['tracking_ratio_pct']:.1f}% | {settled_str} | {overshoot:.1f}% |\n"
    
    # 統計摘要
    if not lin_tracking.empty:
        settled_count = lin_tracking['is_settled'].sum()
        total_count = len(lin_tracking)
        avg_tracking = lin_tracking['tracking_ratio_pct'].mean()
        avg_sse_pct = lin_tracking['sse_percent'].mean()
        
        report += f"""

**線速度追蹤統計摘要**：
- 穩態達成率：{settled_count}/{total_count} ({100*settled_count/total_count:.0f}%)
- 平均追蹤率：{avg_tracking:.1f}%
- 平均穩態誤差：{avg_sse_pct:.2f}%
"""

    report += """

---

## 4. 角速度追蹤性能（嚴格穩態誤差）

| 命令 (rad/s) | 實際 (rad/s) | Ts (ms) | SSE (rad/s) | SSE (%) | RMSE | 追蹤率 (%) | 穩態 |
|--------------|--------------|---------|-------------|---------|------|------------|------|
"""
    
    # 抽樣顯示角速度（每 0.5 rad/s 取樣）
    if not ang_tracking.empty:
        shown_values = []
        for val in np.arange(-4.0, 4.5, 0.5):
            closest = ang_tracking.iloc[(ang_tracking['cmd_ang_z'] - val).abs().argsort()[:1]]
            if len(closest) > 0 and closest.iloc[0]['cmd_ang_z'] not in shown_values:
                row = closest.iloc[0]
                if abs(row['cmd_ang_z'] - val) < 0.3:
                    ts_str = f"{row['settling_time_ms']:.0f}" if row['settling_time_ms'] is not None else "N/A"
                    settled_str = "✓" if row['is_settled'] else "✗"
                    report += f"| {row['cmd_ang_z']:.2f} | {row['actual_mean']:.4f} | {ts_str} | {row['sse_absolute']:.4f} | {row['sse_percent']:.1f}% | {row['rmse']:.4f} | {row['tracking_ratio_pct']:.1f}% | {settled_str} |\n"
                    shown_values.append(row['cmd_ang_z'])
    
    report += f"""

---

## 5. 控制延遲分析

| 指標 | 數值 |
|------|------|
| 平均延遲 | {latency_stats.get('mean_ms', 0):.1f} ms |
| 最小延遲 | {latency_stats.get('min_ms', 0):.1f} ms |
| 最大延遲 | {latency_stats.get('max_ms', 0):.1f} ms |
| 中位數 | {latency_stats.get('median_ms', 0):.1f} ms |
| 標準差 | {latency_stats.get('std_ms', 0):.1f} ms |
| 測量樣本數 | {latency_stats.get('count', 0)} |

**延遲定義**: 從發送命令 (cmd_vel) 到 odometry 速度開始響應的時間差。

---

## 6. 煞車特性分析

共偵測到 **{len(braking_events)}** 次煞車事件。

"""
    
    if braking_events:
        report += "| 初速度 (m/s) | 停止時間 (ms) | 滑行距離 (m) | 平均減速度 (m/s²) |\n"
        report += "|--------------|---------------|--------------|-------------------|\n"
        
        for evt in braking_events[:20]:
            report += f"| {evt['start_speed_mps']:.4f} | {evt['stop_time_ms']:.0f} | {evt['slide_distance_m']:.4f} | {evt['avg_decel_mps2']:.4f} |\n"
        
        if len(braking_events) > 3:
            avg_decel = np.mean([e['avg_decel_mps2'] for e in braking_events])
            avg_slide = np.mean([e['slide_distance_m'] for e in braking_events])
            avg_time = np.mean([e['stop_time_ms'] for e in braking_events])
            
            report += f"""

**煞車統計摘要**：
| 指標 | 數值 |
|------|------|
| 平均減速度 | {avg_decel:.4f} m/s² |
| 平均滑行距離 | {avg_slide:.4f} m |
| 平均停止時間 | {avg_time:.0f} ms |
"""
    
    report += """

---

## 7. 生成的圖表

### 速度追蹤分析
- **velocity_comparison.png** - 命令 vs 實際速度時序對比
- **linear_tracking_performance.png** - 線速度追蹤性能圖（時序 + 追蹤率）
- **angular_tracking_performance.png** - 角速度追蹤性能圖（時序 + 追蹤率）

### 延遲分析
- **latency_analysis.png** - 控制延遲時序圖與 CDF

### 煞車特性分析
- **braking_analysis.png** - 煞車特性綜合分析

---

*報告由 `speed_analysis_v4.py` 自動生成（嚴格穩態誤差版本）*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ Saved: {output_path}")


# ============================================================
# 主程式
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 speed_analysis_v4.py <csv_file>")
        print("Example: python3 speed_analysis_v4.py speed_test_20251215_234034.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # 輸出目錄
    output_dir = csv_path.parent / f"speed_analysis_v4_{csv_path.stem}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"機器人速度測試分析 v4 (嚴格穩態誤差版本)")
    print(f"{'='*60}")
    print(f"Loading: {csv_path}")
    print(f"Tolerance: ±{TOLERANCE_PERCENT*100:.0f}%, Min steady samples: {MIN_STEADY_SAMPLES}")
    
    # 載入數據
    df = load_data(csv_path)
    df.attrs['source'] = str(csv_path)
    
    print(f"Samples: {len(df)}, Duration: {df['time'].max():.1f}s")
    
    # 分析
    print("\n[1/6] Analyzing test overview...")
    overview = analyze_test_overview(df)
    
    print("[2/6] Analyzing speed extremes...")
    extremes = analyze_speed_extremes(df)
    
    print("[3/6] Analyzing tracking performance (strict SSE)...")
    lin_tracking = analyze_linear_tracking_strict(df)
    ang_tracking = analyze_angular_tracking_strict(df)
    
    # 輸出穩態達成統計
    if not lin_tracking.empty:
        settled = lin_tracking['is_settled'].sum()
        total = len(lin_tracking)
        print(f"       Linear: {settled}/{total} commands reached steady state")
    if not ang_tracking.empty:
        settled = ang_tracking['is_settled'].sum()
        total = len(ang_tracking)
        print(f"       Angular: {settled}/{total} commands reached steady state")
    
    print("[4/6] Analyzing latency...")
    latency_stats, latency_ts = analyze_latency(df)
    
    print("[5/6] Analyzing braking characteristics...")
    braking = analyze_braking(df)
    
    # 生成圖表
    print("\n[6/6] Generating visualizations...")
    plot_velocity_comparison(df, output_dir / 'velocity_comparison.png')
    plot_linear_tracking_performance(df, lin_tracking, output_dir / 'linear_tracking_performance.png')
    plot_angular_tracking_performance(df, ang_tracking, output_dir / 'angular_tracking_performance.png')
    plot_latency_analysis(latency_ts, latency_stats, output_dir / 'latency_analysis.png')
    plot_braking_analysis(braking, output_dir / 'braking_analysis.png')
    
    # 生成報告
    print("\n[7/7] Generating report...")
    generate_report(df, output_dir / 'analysis_report_v3.md',
                   overview, extremes, lin_tracking, ang_tracking,
                   latency_stats, braking)
    
    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
