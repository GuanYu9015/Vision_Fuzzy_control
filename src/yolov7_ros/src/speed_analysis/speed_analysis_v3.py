#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機器人速度測試數據分析程式 v2

功能：
1. 線速度/角速度的命令 vs 實際值線圖
2. 所有命令與實際響應的延遲線圖
3. 完整統計表格（測試概覽、速度極值、追蹤性能、煞車特性）

使用方式: python3 speed_analysis_v3.py speed_test_20251215_234034.csv
使用方式: rosrun yolov7_ros speed_analysis_v3.py speed_test_20251215_234034.csv

輸出：
- 圖表：velocity_comparison.png, latency_analysis.png, braking_analysis.png
- 報告：analysis_report_v2.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import sys
from datetime import datetime

# 使用非互動式後端
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_data(filepath: str) -> pd.DataFrame:
    """載入並預處理數據"""
    df = pd.read_csv(filepath)
    # 轉為相對時間 (秒)
    df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    return df


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


def analyze_linear_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """
    線速度追蹤性能分析
    包含穩態誤差 (Steady-State Error) 計算
    """
    results = []
    cmd_values = sorted(df['cmd_lin_x'].unique())
    
    for cmd_val in cmd_values:
        if cmd_val <= 0:
            continue
        
        subset = df[df['cmd_lin_x'] == cmd_val]
        if len(subset) < 5:
            continue
        
        # 取穩態數據（後 70%）
        steady = subset.iloc[int(len(subset) * 0.3):]
        if len(steady) < 3:
            continue
        
        actual_mean = steady['odom_lin_speed'].mean()
        actual_std = steady['odom_lin_speed'].std()
        
        # 穩態誤差 (Steady-State Error) 計算
        sse_absolute = cmd_val - actual_mean  # 絕對穩態誤差
        sse_percent = (sse_absolute / cmd_val) * 100 if cmd_val > 0 else 0  # 相對穩態誤差 (%)
        
        # 計算 RMSE (Root Mean Square Error)
        errors = cmd_val - steady['odom_lin_speed']
        rmse = np.sqrt((errors ** 2).mean())
        
        tracking_ratio = (actual_mean / cmd_val) * 100 if cmd_val > 0 else 0
        
        results.append({
            'cmd_lin_x': round(cmd_val, 2),
            'actual_mean': round(actual_mean, 4),
            'actual_std': round(actual_std, 4),
            'sse_absolute': round(sse_absolute, 4),
            'sse_percent': round(sse_percent, 2),
            'rmse': round(rmse, 4),
            'tracking_ratio_pct': round(tracking_ratio, 1),
            'samples': len(steady)
        })
    
    return pd.DataFrame(results)


def analyze_angular_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """
    角速度追蹤性能分析
    包含穩態誤差 (Steady-State Error) 計算
    """
    results = []
    cmd_values = sorted(df['cmd_ang_z'].unique())
    
    for cmd_val in cmd_values:
        if abs(cmd_val) < 0.1:
            continue
        
        subset = df[df['cmd_ang_z'] == cmd_val]
        if len(subset) < 5:
            continue
        
        steady = subset.iloc[int(len(subset) * 0.3):]
        if len(steady) < 3:
            continue
        
        actual_mean = steady['odom_ang_z'].mean()
        actual_std = steady['odom_ang_z'].std()
        
        # 穩態誤差 (Steady-State Error) 計算
        sse_absolute = cmd_val - actual_mean
        sse_percent = (sse_absolute / cmd_val) * 100 if abs(cmd_val) > 0.01 else 0
        
        # 計算 RMSE
        errors = cmd_val - steady['odom_ang_z']
        rmse = np.sqrt((errors ** 2).mean())
        
        tracking_ratio = (actual_mean / cmd_val) * 100 if abs(cmd_val) > 0.01 else 0
        
        results.append({
            'cmd_ang_z': round(cmd_val, 2),
            'actual_mean': round(actual_mean, 4),
            'actual_std': round(actual_std, 4),
            'sse_absolute': round(sse_absolute, 4),
            'sse_percent': round(sse_percent, 2),
            'rmse': round(rmse, 4),
            'tracking_ratio_pct': round(tracking_ratio, 1),
            'samples': len(steady)
        })
    
    return pd.DataFrame(results)


def analyze_latency(df: pd.DataFrame) -> tuple:
    """
    延遲分析：計算每次命令變化到實際速度響應的延遲
    返回: (延遲統計字典, 延遲時序列表)
    """
    # 偵測命令變化點（線速度或角速度變化 > 0.01）
    df = df.copy()
    df['cmd_lin_change'] = df['cmd_lin_x'].diff().abs() > 0.01
    df['cmd_ang_change'] = df['cmd_ang_z'].diff().abs() > 0.01
    df['cmd_change'] = df['cmd_lin_change'] | df['cmd_ang_change']
    
    latencies = []
    latency_timeseries = []  # 用於繪製線圖
    
    change_indices = df[df['cmd_change']].index.tolist()
    
    for idx in change_indices:
        if idx < 2 or idx > len(df) - 20:
            continue
        
        cmd_time = df.loc[idx, 'timestamp']
        event_time = df.loc[idx, 'time']  # 相對時間
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
                    latency = (df.loc[j, 'timestamp'] - cmd_time) * 1000  # ms
                    if 10 < latency < 500:
                        latencies.append({
                            'type': 'linear_accel',
                            'latency_ms': round(latency, 1),
                            'cmd_step': round(cmd_lin_after - cmd_lin_before, 2),
                            'cmd_before': round(cmd_lin_before, 3),
                            'cmd_after': round(cmd_lin_after, 3),
                            'event_time': round(event_time, 2),
                            'sample_index': idx
                        })
                        latency_timeseries.append({
                            'time': event_time,
                            'latency_ms': round(latency, 1),
                            'type': 'linear',
                            'cmd_before': round(cmd_lin_before, 3),
                            'cmd_after': round(cmd_lin_after, 3),
                            'sample_index': idx
                        })
                    break
        
        # 角速度變化的情況
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
                            'cmd_step': round(cmd_ang_after - cmd_ang_before, 2),
                            'cmd_before': round(cmd_ang_before, 3),
                            'cmd_after': round(cmd_ang_after, 3),
                            'event_time': round(event_time, 2),
                            'sample_index': idx
                        })
                        latency_timeseries.append({
                            'time': event_time,
                            'latency_ms': round(latency, 1),
                            'type': 'angular',
                            'cmd_before': round(cmd_ang_before, 3),
                            'cmd_after': round(cmd_ang_after, 3),
                            'sample_index': idx
                        })
                    break
    
    # 統計
    if latencies:
        lat_values = [l['latency_ms'] for l in latencies]
        min_idx = np.argmin(lat_values)
        max_idx = np.argmax(lat_values)
        
        stats = {
            'mean_ms': round(np.mean(lat_values), 1),
            'min_ms': round(np.min(lat_values), 1),
            'max_ms': round(np.max(lat_values), 1),
            'std_ms': round(np.std(lat_values), 1),
            'median_ms': round(np.median(lat_values), 1),
            'count': len(latencies),
            # 最小延遲詳細資訊
            'min_event': {
                'time': latencies[min_idx]['event_time'],
                'latency_ms': latencies[min_idx]['latency_ms'],
                'type': latencies[min_idx]['type'],
                'cmd_before': latencies[min_idx]['cmd_before'],
                'cmd_after': latencies[min_idx]['cmd_after'],
                'sample_index': latencies[min_idx]['sample_index']
            },
            # 最大延遲詳細資訊
            'max_event': {
                'time': latencies[max_idx]['event_time'],
                'latency_ms': latencies[max_idx]['latency_ms'],
                'type': latencies[max_idx]['type'],
                'cmd_before': latencies[max_idx]['cmd_before'],
                'cmd_after': latencies[max_idx]['cmd_after'],
                'sample_index': latencies[max_idx]['sample_index']
            }
        }
    else:
        stats = {'mean_ms': 0, 'count': 0}
    
    return stats, latency_timeseries


def analyze_braking(df: pd.DataFrame) -> list:
    """
    煞車特性分析
    找出 cmd_lin_x 從正值變為 0 的事件，分析停止特性
    """
    braking_events = []
    
    i = 1
    while i < len(df) - 1:
        # 偵測從移動到停止命令的轉換
        prev_cmd = df.loc[i - 1, 'cmd_lin_x']
        curr_cmd = df.loc[i, 'cmd_lin_x']
        prev_speed = df.loc[i - 1, 'odom_lin_speed']
        
        if prev_cmd > 0.05 and curr_cmd == 0 and prev_speed > 0.05:
            start_idx = i
            start_speed = prev_speed
            start_time = df.loc[i, 'timestamp']
            start_x = df.loc[i, 'odom_x']
            start_y = df.loc[i, 'odom_y']
            
            # 找到完全停止點 (速度 < 0.005 m/s)
            stop_found = False
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
                    stop_found = True
                    i = j
                    break
            
            if not stop_found:
                i += 1
        else:
            i += 1
    
    return braking_events


def plot_velocity_comparison(df: pd.DataFrame, output_path: Path):
    """繪製線速度和角速度的命令 vs 實際值線圖"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # --- 線速度對比 ---
    ax1 = axes[0]
    ax1.plot(df['time'], df['cmd_lin_x'], 'b-', label='Command (cmd_lin_x)', 
             alpha=0.8, linewidth=1.2)
    ax1.plot(df['time'], df['odom_lin_speed'], 'r-', label='Actual (odom_lin_speed)', 
             alpha=0.7, linewidth=1.0)
    ax1.set_ylabel('Linear Velocity (m/s)', fontsize=12)
    ax1.set_title('Linear Velocity: Command vs Actual', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(df['cmd_lin_x'].min(), -0.1) - 0.1, 
                  max(df['cmd_lin_x'].max(), df['odom_lin_speed'].max()) + 0.1])
    
    # 填充誤差區域
    ax1.fill_between(df['time'], df['cmd_lin_x'], df['odom_lin_speed'], 
                     alpha=0.2, color='purple', label='Tracking Error')
    
    # --- 角速度對比 ---
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


def plot_tracking_error(df: pd.DataFrame, lin_tracking: pd.DataFrame, 
                        ang_tracking: pd.DataFrame, output_dir: Path):
    """
    繪製追蹤誤差視覺化圖表
    包含：時序誤差圖、穩態誤差分析圖、追蹤率比較圖
    """
    # ===== 1. 線速度追蹤誤差時序圖 (獨立圖) =====
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    # 計算瞬時誤差
    lin_error = df['cmd_lin_x'] - df['odom_lin_speed']
    ax1.plot(df['time'], lin_error, 'b-', alpha=0.7, linewidth=0.8, label='Tracking Error')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=lin_error.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean Error: {lin_error.mean():.4f} m/s')
    ax1.fill_between(df['time'], 0, lin_error, alpha=0.3, color='blue')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Error (m/s)', fontsize=12)
    ax1.set_title('Linear Velocity Tracking Error Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'tracking_error_linear_timeseries.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'tracking_error_linear_timeseries.png'}")
    
    # ===== 2. 角速度追蹤誤差時序圖 (獨立圖) =====
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ang_error = df['cmd_ang_z'] - df['odom_ang_z']
    ax2.plot(df['time'], ang_error, 'g-', alpha=0.7, linewidth=0.8, label='Tracking Error')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=ang_error.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean Error: {ang_error.mean():.4f} rad/s')
    ax2.fill_between(df['time'], 0, ang_error, alpha=0.3, color='green')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Error (rad/s)', fontsize=12)
    ax2.set_title('Angular Velocity Tracking Error Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'tracking_error_angular_timeseries.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'tracking_error_angular_timeseries.png'}")
    
    # ===== 3. 穩態誤差 vs 命令速度圖 (獨立圖) =====
    if not lin_tracking.empty:
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        
        # 線速度穩態誤差
        ax = axes3[0]
        cmd_vals = lin_tracking['cmd_lin_x'].values
        sse_vals = lin_tracking['sse_absolute'].values
        sse_pct = lin_tracking['sse_percent'].values
        
        # 雙 Y 軸：絕對誤差 + 相對誤差
        color1 = 'tab:blue'
        ax.set_xlabel('Command Linear Velocity (m/s)', fontsize=11)
        ax.set_ylabel('Absolute SSE (m/s)', color=color1, fontsize=11)
        bars = ax.bar(cmd_vals - 0.015, sse_vals, width=0.03, alpha=0.7, color=color1, 
                      label='Absolute SSE')
        ax.tick_params(axis='y', labelcolor=color1)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        
        ax2_twin = ax.twinx()
        color2 = 'tab:orange'
        ax2_twin.set_ylabel('Relative SSE (%)', color=color2, fontsize=11)
        ax2_twin.plot(cmd_vals, sse_pct, 'o-', color=color2, linewidth=2, markersize=6,
                     label='Relative SSE %')
        ax2_twin.tick_params(axis='y', labelcolor=color2)
        
        ax.set_title('Linear Velocity: Steady-State Error Analysis', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 角速度穩態誤差
        if not ang_tracking.empty:
            ax = axes3[1]
            cmd_vals_ang = ang_tracking['cmd_ang_z'].values
            sse_vals_ang = ang_tracking['sse_absolute'].values
            sse_pct_ang = ang_tracking['sse_percent'].values
            
            ax.set_xlabel('Command Angular Velocity (rad/s)', fontsize=11)
            ax.set_ylabel('Absolute SSE (rad/s)', color=color1, fontsize=11)
            ax.bar(cmd_vals_ang, sse_vals_ang, width=0.08, alpha=0.7, color=color1)
            ax.tick_params(axis='y', labelcolor=color1)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            ax2_twin = ax.twinx()
            ax2_twin.set_ylabel('Relative SSE (%)', color=color2, fontsize=11)
            ax2_twin.plot(cmd_vals_ang, sse_pct_ang, 'o-', color=color2, linewidth=2, markersize=6)
            ax2_twin.tick_params(axis='y', labelcolor=color2)
            
            ax.set_title('Angular Velocity: Steady-State Error Analysis', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tracking_sse_analysis.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: {output_dir / 'tracking_sse_analysis.png'}")
    
    # ===== 4. 追蹤率與 RMSE 比較圖 (獨立圖) =====
    if not lin_tracking.empty:
        fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
        
        # 線速度追蹤率
        ax = axes4[0]
        cmd_vals = lin_tracking['cmd_lin_x'].values
        tracking_ratio = lin_tracking['tracking_ratio_pct'].values
        rmse_vals = lin_tracking['rmse'].values
        
        color1 = 'tab:green'
        ax.set_xlabel('Command Linear Velocity (m/s)', fontsize=11)
        ax.set_ylabel('Tracking Ratio (%)', color=color1, fontsize=11)
        ax.bar(cmd_vals - 0.015, tracking_ratio, width=0.03, alpha=0.7, color=color1)
        ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, label='Ideal (100%)')
        ax.tick_params(axis='y', labelcolor=color1)
        ax.set_ylim([min(tracking_ratio) - 10, max(tracking_ratio) + 10])
        
        ax2_twin = ax.twinx()
        color2 = 'tab:red'
        ax2_twin.set_ylabel('RMSE (m/s)', color=color2, fontsize=11)
        ax2_twin.plot(cmd_vals, rmse_vals, 's-', color=color2, linewidth=2, markersize=6)
        ax2_twin.tick_params(axis='y', labelcolor=color2)
        
        ax.set_title('Linear Velocity: Tracking Ratio & RMSE', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 角速度追蹤率與 RMSE
        if not ang_tracking.empty:
            ax = axes4[1]
            cmd_vals_ang = ang_tracking['cmd_ang_z'].values
            tracking_ratio_ang = ang_tracking['tracking_ratio_pct'].values
            rmse_vals_ang = ang_tracking['rmse'].values
            
            ax.set_xlabel('Command Angular Velocity (rad/s)', fontsize=11)
            ax.set_ylabel('Tracking Ratio (%)', color=color1, fontsize=11)
            ax.bar(cmd_vals_ang, tracking_ratio_ang, width=0.08, alpha=0.7, color=color1)
            ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, label='Ideal (100%)')
            ax.tick_params(axis='y', labelcolor=color1)
            ax.set_ylim([min(tracking_ratio_ang) - 10, max(tracking_ratio_ang) + 10])
            
            ax2_twin = ax.twinx()
            ax2_twin.set_ylabel('RMSE (rad/s)', color=color2, fontsize=11)
            ax2_twin.plot(cmd_vals_ang, rmse_vals_ang, 's-', color=color2, linewidth=2, markersize=6)
            ax2_twin.tick_params(axis='y', labelcolor=color2)
            
            ax.set_title('Angular Velocity: Tracking Ratio & RMSE', fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tracking_ratio_rmse.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: {output_dir / 'tracking_ratio_rmse.png'}")
    
    # ===== 5. 追蹤誤差組合圖 =====
    fig5, axes5 = plt.subplots(2, 2, figsize=(16, 10))
    
    # 線速度誤差時序
    ax = axes5[0, 0]
    lin_error = df['cmd_lin_x'] - df['odom_lin_speed']
    ax.plot(df['time'], lin_error, 'b-', alpha=0.6, linewidth=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=lin_error.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {lin_error.mean():.4f}')
    ax.fill_between(df['time'], 0, lin_error, alpha=0.2, color='blue')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Error (m/s)', fontsize=10)
    ax.set_title('Linear Velocity Tracking Error', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 角速度誤差時序
    ax = axes5[0, 1]
    ang_error = df['cmd_ang_z'] - df['odom_ang_z']
    ax.plot(df['time'], ang_error, 'g-', alpha=0.6, linewidth=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=ang_error.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {ang_error.mean():.4f}')
    ax.fill_between(df['time'], 0, ang_error, alpha=0.2, color='green')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Error (rad/s)', fontsize=10)
    ax.set_title('Angular Velocity Tracking Error', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 線速度 SSE 條形圖
    ax = axes5[1, 0]
    if not lin_tracking.empty:
        cmd_vals = lin_tracking['cmd_lin_x'].values
        sse_pct = lin_tracking['sse_percent'].values
        colors = ['green' if abs(v) < 5 else 'orange' if abs(v) < 10 else 'red' for v in sse_pct]
        ax.bar(cmd_vals, sse_pct, width=0.03, alpha=0.7, color=colors, edgecolor='black')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axhline(y=5, color='green', linestyle=':', alpha=0.5, label='±5% threshold')
        ax.axhline(y=-5, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel('Command (m/s)', fontsize=10)
    ax.set_ylabel('SSE (%)', fontsize=10)
    ax.set_title('Linear Velocity Steady-State Error %', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 角速度 SSE 條形圖
    ax = axes5[1, 1]
    if not ang_tracking.empty:
        cmd_vals_ang = ang_tracking['cmd_ang_z'].values
        sse_pct_ang = ang_tracking['sse_percent'].values
        colors = ['green' if abs(v) < 5 else 'orange' if abs(v) < 10 else 'red' for v in sse_pct_ang]
        ax.bar(cmd_vals_ang, sse_pct_ang, width=0.08, alpha=0.7, color=colors, edgecolor='black')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axhline(y=5, color='green', linestyle=':', alpha=0.5, label='±5% threshold')
        ax.axhline(y=-5, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel('Command (rad/s)', fontsize=10)
    ax.set_ylabel('SSE (%)', fontsize=10)
    ax.set_title('Angular Velocity Steady-State Error %', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tracking_error_analysis.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'tracking_error_analysis.png'}")


def plot_latency_timeseries(latency_data: list, latency_stats: dict, output_path: Path):
    """繪製所有命令與實際響應的延遲線圖，並標註極值點"""
    if not latency_data:
        print(f"  ⚠ No latency data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 分離線速度和角速度延遲
    lin_data = [d for d in latency_data if d['type'] == 'linear']
    ang_data = [d for d in latency_data if d['type'] == 'angular']
    all_latencies = [d['latency_ms'] for d in latency_data]
    
    # 取得極值資訊
    min_event = latency_stats.get('min_event', {})
    max_event = latency_stats.get('max_event', {})
    
    # --- 線速度延遲時序 ---
    ax1 = axes[0, 0]
    if lin_data:
        times = [d['time'] for d in lin_data]
        lats = [d['latency_ms'] for d in lin_data]
        ax1.plot(times, lats, 'bo-', markersize=5, alpha=0.7, linewidth=1, label='Linear Latency')
        ax1.axhline(y=np.mean(lats), color='b', linestyle='--', alpha=0.5, 
                    label=f'Mean: {np.mean(lats):.1f} ms')
        
        # 標註最小/最大延遲點（如果是線速度類型）
        if min_event.get('type') == 'linear_accel':
            ax1.scatter([min_event['time']], [min_event['latency_ms']], 
                       c='green', s=150, marker='v', zorder=5, 
                       label=f"Min: {min_event['latency_ms']} ms @ {min_event['time']}s")
            ax1.annotate(f"MIN\n{min_event['latency_ms']}ms", 
                        (min_event['time'], min_event['latency_ms']),
                        textcoords='offset points', xytext=(10, -20), fontsize=9, color='green')
        if max_event.get('type') == 'linear_accel':
            ax1.scatter([max_event['time']], [max_event['latency_ms']], 
                       c='red', s=150, marker='^', zorder=5,
                       label=f"Max: {max_event['latency_ms']} ms @ {max_event['time']}s")
            ax1.annotate(f"MAX\n{max_event['latency_ms']}ms", 
                        (max_event['time'], max_event['latency_ms']),
                        textcoords='offset points', xytext=(10, 10), fontsize=9, color='red')
    ax1.set_ylabel('Latency (ms)', fontsize=11)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_title('Linear Velocity Command → Response Latency', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(all_latencies) + 50])
    
    # --- 角速度延遲時序 ---
    ax2 = axes[0, 1]
    if ang_data:
        times = [d['time'] for d in ang_data]
        lats = [d['latency_ms'] for d in ang_data]
        ax2.plot(times, lats, 'go-', markersize=5, alpha=0.7, linewidth=1, label='Angular Latency')
        ax2.axhline(y=np.mean(lats), color='g', linestyle='--', alpha=0.5, 
                    label=f'Mean: {np.mean(lats):.1f} ms')
        
        if min_event.get('type') == 'angular_accel':
            ax2.scatter([min_event['time']], [min_event['latency_ms']], 
                       c='green', s=150, marker='v', zorder=5,
                       label=f"Min: {min_event['latency_ms']} ms @ {min_event['time']}s")
            ax2.annotate(f"MIN\n{min_event['latency_ms']}ms", 
                        (min_event['time'], min_event['latency_ms']),
                        textcoords='offset points', xytext=(10, -20), fontsize=9, color='green')
        if max_event.get('type') == 'angular_accel':
            ax2.scatter([max_event['time']], [max_event['latency_ms']], 
                       c='red', s=150, marker='^', zorder=5,
                       label=f"Max: {max_event['latency_ms']} ms @ {max_event['time']}s")
            ax2.annotate(f"MAX\n{max_event['latency_ms']}ms", 
                        (max_event['time'], max_event['latency_ms']),
                        textcoords='offset points', xytext=(10, 10), fontsize=9, color='red')
    ax2.set_ylabel('Latency (ms)', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_title('Angular Velocity Command → Response Latency', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- 延遲分佈直方圖 ---
    ax3 = axes[1, 0]
    ax3.hist(all_latencies, bins=25, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(x=latency_stats['mean_ms'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {latency_stats['mean_ms']} ms")
    ax3.axvline(x=latency_stats['median_ms'], color='orange', linestyle='--', linewidth=2,
                label=f"Median: {latency_stats['median_ms']} ms")
    ax3.axvline(x=latency_stats['min_ms'], color='green', linestyle=':', linewidth=2,
                label=f"Min: {latency_stats['min_ms']} ms")
    ax3.axvline(x=latency_stats['max_ms'], color='purple', linestyle=':', linewidth=2,
                label=f"Max: {latency_stats['max_ms']} ms")
    ax3.set_xlabel('Latency (ms)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Latency Distribution', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # --- 累積分佈函數 (CDF) ---
    ax4 = axes[1, 1]
    sorted_lat = np.sort(all_latencies)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat) * 100
    ax4.plot(sorted_lat, cdf, 'b-', linewidth=2)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax4.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95%')
    ax4.axhline(y=99, color='gray', linestyle='--', alpha=0.5, label='99%')
    # 標注百分位數
    p50 = np.percentile(all_latencies, 50)
    p95 = np.percentile(all_latencies, 95)
    p99 = np.percentile(all_latencies, 99)
    ax4.scatter([p50, p95, p99], [50, 95, 99], c='red', s=50, zorder=5)
    ax4.annotate(f'{p50:.0f}ms', (p50, 50), textcoords='offset points', xytext=(5, 5), fontsize=9)
    ax4.annotate(f'{p95:.0f}ms', (p95, 95), textcoords='offset points', xytext=(5, -10), fontsize=9)
    ax4.annotate(f'{p99:.0f}ms', (p99, 99), textcoords='offset points', xytext=(5, -10), fontsize=9)
    ax4.set_xlabel('Latency (ms)', fontsize=11)
    ax4.set_ylabel('Cumulative %', fontsize=11)
    ax4.set_title('Latency CDF (Cumulative Distribution)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_braking_analysis(braking_events: list, output_dir: Path):
    """
    繪製煞車分析圖，包含減速度 CDF
    每張子圖獨立儲存為單獨檔案
    """
    if not braking_events:
        print(f"  ⚠ No braking events to plot")
        return
    
    speeds = [b['start_speed_mps'] for b in braking_events]
    distances = [b['slide_distance_m'] for b in braking_events]
    times = [b['stop_time_ms'] for b in braking_events]
    decels = [b['avg_decel_mps2'] for b in braking_events]
    
    # ===== 1. 初速 vs 滑行距離 (獨立圖) =====
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(speeds, distances, alpha=0.7, s=50, c='blue', edgecolors='black', linewidths=0.5)
    if len(speeds) > 2:
        z = np.polyfit(speeds, distances, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(speeds), max(speeds), 100)
        ax1.plot(x_fit, p(x_fit), 'r--', alpha=0.8, linewidth=2,
                label=f'Linear Fit: d = {z[0]:.4f}v + {z[1]:.4f}')
        ax1.legend(fontsize=10)
    ax1.set_xlabel('Initial Speed (m/s)', fontsize=12)
    ax1.set_ylabel('Sliding Distance (m)', fontsize=12)
    ax1.set_title('Braking Analysis: Speed vs Sliding Distance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # 標註統計資訊
    stats_text = f'n = {len(speeds)}\nMean Distance: {np.mean(distances):.4f} m'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_dir / 'braking_speed_vs_distance.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'braking_speed_vs_distance.png'}")
    
    # ===== 2. 初速 vs 停止時間 (獨立圖) =====
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(speeds, times, alpha=0.7, s=50, c='green', edgecolors='black', linewidths=0.5)
    ax2.set_xlabel('Initial Speed (m/s)', fontsize=12)
    ax2.set_ylabel('Stop Time (ms)', fontsize=12)
    ax2.set_title('Braking Analysis: Speed vs Stop Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    stats_text = f'n = {len(times)}\nMean Time: {np.mean(times):.1f} ms'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_dir / 'braking_speed_vs_time.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'braking_speed_vs_time.png'}")
    
    # ===== 3. 減速度分佈直方圖 (獨立圖) =====
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    n, bins, patches = ax3.hist(decels, bins=15, alpha=0.7, color='orange', edgecolor='black')
    mean_decel = np.mean(decels)
    std_decel = np.std(decels)
    ax3.axvline(x=mean_decel, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_decel:.4f} m/s²')
    ax3.axvline(x=mean_decel - std_decel, color='purple', linestyle=':', linewidth=1.5,
                label=f'Mean ± Std: {std_decel:.4f}')
    ax3.axvline(x=mean_decel + std_decel, color='purple', linestyle=':', linewidth=1.5)
    ax3.set_xlabel('Deceleration (m/s²)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Braking Analysis: Deceleration Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'braking_decel_histogram.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'braking_decel_histogram.png'}")
    
    # ===== 4. 減速度 CDF (獨立圖) =====
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sorted_decels = np.sort(decels)
    cdf = np.arange(1, len(sorted_decels) + 1) / len(sorted_decels) * 100
    ax4.plot(sorted_decels, cdf, 'b-', linewidth=2, label='CDF')
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=95, color='gray', linestyle='--', alpha=0.5)
    # 標註百分位數
    p50 = np.percentile(decels, 50)
    p95 = np.percentile(decels, 95)
    ax4.scatter([p50, p95], [50, 95], c='red', s=80, zorder=5, marker='o')
    ax4.annotate(f'P50: {p50:.4f} m/s²', (p50, 50), textcoords='offset points', 
                 xytext=(10, 5), fontsize=10, color='red')
    ax4.annotate(f'P95: {p95:.4f} m/s²', (p95, 95), textcoords='offset points', 
                 xytext=(10, -15), fontsize=10, color='red')
    ax4.set_xlabel('Deceleration (m/s²)', fontsize=12)
    ax4.set_ylabel('Cumulative %', fontsize=12)
    ax4.set_title('Braking Analysis: Deceleration CDF', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(output_dir / 'braking_decel_cdf.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'braking_decel_cdf.png'}")
    
    # ===== 5. 組合圖 (保留相容性) =====
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
    ax.plot(sorted_decels, cdf, 'b-', linewidth=2)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5)
    ax.scatter([p50, p95], [50, 95], c='red', s=60, zorder=5)
    ax.annotate(f'{p50:.3f}', (p50, 50), textcoords='offset points', xytext=(5, 5), fontsize=9)
    ax.annotate(f'{p95:.3f}', (p95, 95), textcoords='offset points', xytext=(5, -10), fontsize=9)
    ax.set_xlabel('Deceleration (m/s²)', fontsize=11)
    ax.set_ylabel('Cumulative %', fontsize=11)
    ax.set_title('Deceleration CDF', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'braking_analysis.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: {output_dir / 'braking_analysis.png'}")


def generate_report(df: pd.DataFrame, output_path: Path, 
                    overview: dict, extremes: dict, 
                    lin_tracking: pd.DataFrame, ang_tracking: pd.DataFrame,
                    latency_stats: dict, braking_events: list):
    """生成完整 Markdown 報告"""
    
    report = f"""# 機器人速度測試分析報告 v2

**生成時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**數據來源**: {df.attrs.get('source', 'speed_test.csv')}

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

## 3. 線速度追蹤性能

| 命令 (m/s) | 實際 (m/s) | 標準差 | SSE (m/s) | SSE (%) | RMSE | 追蹤率 (%) | 樣本數 |
|------------|------------|--------|-----------|---------|------|------------|--------|
"""
    
    for _, row in lin_tracking.iterrows():
        report += f"| {row['cmd_lin_x']:.2f} | {row['actual_mean']:.4f} | {row['actual_std']:.4f} | {row['sse_absolute']:.4f} | {row['sse_percent']:.2f}% | {row['rmse']:.4f} | {row['tracking_ratio_pct']:.1f}% | {row['samples']} |\n"
    
    report += """

**穩態誤差說明**: SSE (Steady-State Error) = 命令速度 - 實際速度。正值表示實際速度不足，負值表示超調。
**RMSE**: Root Mean Square Error，穩態期間的均方根誤差，反映追蹤穩定性。

---

## 4. 角速度追蹤性能

| 命令 (rad/s) | 實際 (rad/s) | 標準差 | SSE (rad/s) | SSE (%) | RMSE | 追蹤率 (%) | 樣本數 |
|--------------|--------------|--------|-------------|---------|------|------------|--------|
"""
    
    # 只顯示部分角速度（每 0.5 rad/s 取樣）
    if not ang_tracking.empty:
        shown_values = []
        for val in np.arange(-4.0, 4.5, 0.5):
            closest = ang_tracking.iloc[(ang_tracking['cmd_ang_z'] - val).abs().argsort()[:1]]
            if len(closest) > 0 and closest.iloc[0]['cmd_ang_z'] not in shown_values:
                row = closest.iloc[0]
                if abs(row['cmd_ang_z'] - val) < 0.3:
                    report += f"| {row['cmd_ang_z']:.2f} | {row['actual_mean']:.4f} | {row['actual_std']:.4f} | {row['sse_absolute']:.4f} | {row['sse_percent']:.2f}% | {row['rmse']:.4f} | {row['tracking_ratio_pct']:.1f}% | {row['samples']} |\n"
                    shown_values.append(row['cmd_ang_z'])
    
    # 延遲極值詳細資訊
    min_event = latency_stats.get('min_event', {})
    max_event = latency_stats.get('max_event', {})
    
    report += f"""

---

## 5. 控制延遲分析

### 統計摘要

| 指標 | 數值 |
|------|------|
| 平均延遲 | {latency_stats.get('mean_ms', 0):.1f} ms |
| 最小延遲 | {latency_stats.get('min_ms', 0):.1f} ms |
| 最大延遲 | {latency_stats.get('max_ms', 0):.1f} ms |
| 中位數 | {latency_stats.get('median_ms', 0):.1f} ms |
| 標準差 | {latency_stats.get('std_ms', 0):.1f} ms |
| 測量樣本數 | {latency_stats.get('count', 0)} |

### 最小延遲事件

| 項目 | 數值 |
|------|------|
| 延遲 | {min_event.get('latency_ms', 0)} ms |
| 發生時間 | {min_event.get('time', 0)} 秒 |
| 類型 | {min_event.get('type', 'N/A')} |
| 命令變化 | {min_event.get('cmd_before', 0)} → {min_event.get('cmd_after', 0)} |
| 樣本索引 | {min_event.get('sample_index', 0)} |

### 最大延遲事件

| 項目 | 數值 |
|------|------|
| 延遲 | {max_event.get('latency_ms', 0)} ms |
| 發生時間 | {max_event.get('time', 0)} 秒 |
| 類型 | {max_event.get('type', 'N/A')} |
| 命令變化 | {max_event.get('cmd_before', 0)} → {max_event.get('cmd_after', 0)} |
| 樣本索引 | {max_event.get('sample_index', 0)} |

**延遲定義**: 從發送命令 (cmd_vel) 到 odometry 速度開始響應的時間差。

---

## 6. 煞車特性分析

共偵測到 **{len(braking_events)}** 次煞車事件。

"""
    
    if braking_events:
        report += "| 初速度 (m/s) | 停止時間 (ms) | 滑行距離 (m) | 平均減速度 (m/s²) |\n"
        report += "|--------------|---------------|--------------|-------------------|\n"
        
        for evt in braking_events[:150]:  # 顯示前 20 筆
            report += f"| {evt['start_speed_mps']:.4f} | {evt['stop_time_ms']:.0f} | {evt['slide_distance_m']:.4f} | {evt['avg_decel_mps2']:.4f} |\n"
        
        if len(braking_events) > 3:
            avg_decel = np.mean([e['avg_decel_mps2'] for e in braking_events])
            avg_slide = np.mean([e['slide_distance_m'] for e in braking_events])
            avg_time = np.mean([e['stop_time_ms'] for e in braking_events])
            
            report += f"""

### 煞車統計摘要

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
- **tracking_error_analysis.png** - 追蹤誤差綜合分析 (組合圖)
- **tracking_error_linear_timeseries.png** - 線速度追蹤誤差時序圖
- **tracking_error_angular_timeseries.png** - 角速度追蹤誤差時序圖
- **tracking_sse_analysis.png** - 穩態誤差 (SSE) 分析圖
- **tracking_ratio_rmse.png** - 追蹤率與 RMSE 比較圖

### 延遲分析
- **latency_analysis.png** - 控制延遲時序圖與 CDF

### 煞車特性分析
- **braking_analysis.png** - 煞車特性綜合分析 (組合圖)
- **braking_speed_vs_distance.png** - 初速 vs 滑行距離
- **braking_speed_vs_time.png** - 初速 vs 停止時間
- **braking_decel_histogram.png** - 減速度分佈直方圖
- **braking_decel_cdf.png** - 減速度累積分佈函數 (CDF)

---

*報告由 `speed_analysis_v2.py` 自動生成*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 speed_analysis_v2.py <csv_file>")
        print("Example: python3 speed_analysis_v2.py speed_test_20251210_003549.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # 輸出目錄
    output_dir = csv_path.parent / f"speed_analysis_v2_{csv_path.stem}"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"機器人速度測試分析 v2")
    print(f"{'='*60}")
    print(f"Loading: {csv_path}")
    
    # 載入數據
    df = load_data(csv_path)
    df.attrs['source'] = str(csv_path)
    
    print(f"Samples: {len(df)}, Duration: {df['time'].max():.1f}s")
    
    # 分析
    print("\n[1/6] Analyzing test overview...")
    overview = analyze_test_overview(df)
    
    print("[2/6] Analyzing speed extremes...")
    extremes = analyze_speed_extremes(df)
    
    print("[3/6] Analyzing tracking performance...")
    lin_tracking = analyze_linear_tracking(df)
    ang_tracking = analyze_angular_tracking(df)
    
    print("[4/6] Analyzing latency...")
    latency_stats, latency_ts = analyze_latency(df)
    
    print("[5/6] Analyzing braking characteristics...")
    braking = analyze_braking(df)
    
    # 生成圖表
    print("\n[6/7] Generating visualizations...")
    plot_velocity_comparison(df, output_dir / 'velocity_comparison.png')
    plot_tracking_error(df, lin_tracking, ang_tracking, output_dir)
    plot_latency_timeseries(latency_ts, latency_stats, output_dir / 'latency_analysis.png')
    plot_braking_analysis(braking, output_dir)
    
    # 生成報告
    print("\n[7/7] Generating report...")
    generate_report(df, output_dir / 'analysis_report_v2.md',
                   overview, extremes, lin_tracking, ang_tracking,
                   latency_stats, braking)
    
    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
