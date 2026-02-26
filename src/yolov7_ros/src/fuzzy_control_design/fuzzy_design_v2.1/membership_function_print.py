#!/usr/bin/python3
"""
模糊控制隸屬函數繪圖程式

依據 mod_fuzzy_control4.py 中定義的隸屬函數參數繪製所有輸入/輸出函數圖。

版本：v2.1 (2026-01-08)
更新內容：
  - E_L 使用非對稱死區設計（ZO 與 NS/PS 分離）
  - V 輸出值壓縮低速區間
  - OMEGA 輸出值降低以減少擺動
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def plot_triangular_mf(ax, title, x_range, mf_dict, labels_order, colors):
    """
    繪製三角形/梯形隸屬函數
    
    Args:
        ax: matplotlib axes
        title: 圖表標題
        x_range: X 軸範圍 (min, max)
        mf_dict: 隸屬函數定義 {label: (a, b, c)}
        labels_order: 標籤順序列表
        colors: 顏色列表
    """
    x = np.linspace(x_range[0], x_range[1], 500)
    
    def trimf(x, a, b, c):
        """三角形/梯形隸屬函數"""
        result = np.zeros_like(x)
        
        # 左上升段
        if b > a:
            mask = (x >= a) & (x <= b)
            result[mask] = (x[mask] - a) / (b - a)
        else:
            result[x <= b] = 1.0
        
        # 右下降段
        if c > b:
            mask = (x >= b) & (x <= c)
            result[mask] = (c - x[mask]) / (c - b)
        else:
            result[x >= b] = 1.0
        
        # 處理梯形情況
        if a == b:
            result[x <= a] = 1.0
        if b == c:
            result[x >= c] = 1.0
            
        return np.clip(result, 0, 1)
    
    for i, label in enumerate(labels_order):
        if label in mf_dict:
            a, b, c = mf_dict[label]
            y = trimf(x, a, b, c)
            ax.plot(x, y, label=label, color=colors[i], linewidth=2)
            ax.fill_between(x, y, alpha=0.1, color=colors[i])
    
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Membership', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(x_range[0], x_range[1])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10)
    
    # 設定 X 軸刻度為關鍵點
    key_points = sorted(set([p for params in mf_dict.values() for p in params]))
    ax.set_xticks(key_points)
    ax.tick_params(axis='x', rotation=45)


def plot_singleton_mf(ax, title, x_range, singleton_dict, labels_order, colors):
    """
    繪製單值型隸屬函數 (Singleton)
    
    Args:
        ax: matplotlib axes
        title: 圖表標題
        x_range: X 軸範圍 (min, max)
        singleton_dict: 單值定義 {label: value}
        labels_order: 標籤順序列表
        colors: 顏色列表
    """
    for i, label in enumerate(labels_order):
        if label in singleton_dict:
            value = singleton_dict[label]
            ax.vlines(x=value, ymin=0, ymax=1, colors=colors[i], linewidth=3, label=f"{label}={value}")
            ax.plot(value, 1, 'o', color=colors[i], markersize=8)
    
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Membership', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(x_range[0], x_range[1])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    
    # 設定 X 軸刻度
    values = sorted(singleton_dict.values())
    ticks = sorted(set([x_range[0]] + values + [x_range[1]]))
    ax.set_xticks(ticks)


# =============================================================================
# 隸屬函數定義（與 mod_fuzzy_control4.py 同步）
# =============================================================================

# 前方距離誤差 e_d [0, 2.08]
E_D = {
    'VN': (0.0, 0.0, 0.52),
    'N': (0.0, 0.52, 1.04),
    'M': (0.52, 1.04, 1.56),
    'F': (1.04, 1.56, 2.08),
    'VF': (1.56, 2.08, 2.08)
}

# 前方距離變化率 e_d_dot [-2.08, 2.08]
E_D_DOT = {
    'NB': (-2.08, -2.08, -1.04),
    'NS': (-2.08, -1.04, 0.0),
    'ZO': (-1.04, 0.0, 1.04),
    'PS': (0.0, 1.04, 2.08),
    'PB': (1.04, 2.08, 2.08)
}

# 橫向誤差 e_l [-0.5, 0.5] - 非對稱死區設計
E_L = {
    'NB': (-0.5, -0.5, -0.15),   # 左肩形梯形 (x < -0.5 為 1)
    'NS': (-0.5, -0.15, -0.05),  # 終點 -0.05
    'ZO': (-0.08, 0.0, 0.08),    # 擴大死區 ±0.08m
    'PS': (0.05, 0.15, 0.5),     # 起點 0.05
    'PB': (0.15, 0.5, 0.5)       # 右肩形梯形 (x > 0.5 為 1)
}

# 橫向變化率 e_l_dot [-1.0, 1.0]
E_L_DOT = {
    'NB': (-1.0, -1.0, -0.5),    # 左肩形梯形
    'NS': (-1.0, -0.5, 0.0),
    'ZO': (-0.5, 0.0, 0.5),
    'PS': (0.0, 0.5, 1.0),
    'PB': (0.5, 1.0, 1.0)        # 右肩形梯形
}

# 線速度輸出 v (Singleton) - 壓縮低速區間
V = {
    'S': 0.0,
    'VS': 0.18,
    'SL': 0.32,
    'M': 0.45,
    'F': 0.55
}

# 角速度輸出 omega (Singleton) - 降低 NS/PS 減少擺動
OMEGA = {
    'NB': -1.4,
    'NS': -0.5,
    'ZO': 0.0,
    'PS': 0.5,
    'PB': 1.4
}


# =============================================================================
# 主程式
# =============================================================================
if __name__ == '__main__':
    # 設定畫布
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 顏色定義
    colors_5 = ['red', 'orange', 'green', 'blue', 'purple']
    
    # =========================================================================
    # 1. Forward Distance Error (前方距離誤差)
    # =========================================================================
    plot_triangular_mf(
        axes[0, 0], 
        'Input 1: Forward Distance Error e_d (m)',
        (-0.2, 2.3),
        E_D,
        ['VN', 'N', 'M', 'F', 'VF'],
        colors_5
    )
    
    # =========================================================================
    # 2. Forward Distance Error Rate (前方距離誤差變化率)
    # =========================================================================
    plot_triangular_mf(
        axes[0, 1],
        'Input 2: Forward Distance Error Rate e_d_dot',
        (-2.5, 2.5),
        E_D_DOT,
        ['NB', 'NS', 'ZO', 'PS', 'PB'],
        colors_5
    )
    
    # =========================================================================
    # 3. Lateral Error (橫向誤差) - 非對稱死區設計
    # =========================================================================
    plot_triangular_mf(
        axes[1, 0],
        'Input 3: Lateral Error e_l (m)',
        (-0.6, 0.6),
        E_L,
        ['NB', 'NS', 'ZO', 'PS', 'PB'],
        colors_5
    )
    
    # =========================================================================
    # 4. Lateral Error Rate (橫向誤差變化率)
    # =========================================================================
    plot_triangular_mf(
        axes[1, 1],
        'Input 4: Lateral Error Rate e_l_dot',
        (-1.25, 1.25),
        E_L_DOT,
        ['NB', 'NS', 'ZO', 'PS', 'PB'],
        colors_5
    )
    
    # =========================================================================
    # 5. Output: Linear Velocity (線速度輸出)
    # =========================================================================
    plot_singleton_mf(
        axes[2, 0],
        'Output 1: Linear Velocity v (m/s)',
        (-0.05, 0.6),
        V,
        ['S', 'VS', 'SL', 'M', 'F'],
        colors_5
    )
    
    # =========================================================================
    # 6. Output: Angular Velocity (角速度輸出)
    # =========================================================================
    plot_singleton_mf(
        axes[2, 1],
        'Output 2: Angular Velocity ω (rad/s)',
        (-1.6, 1.6),
        OMEGA,
        ['NB', 'NS', 'ZO', 'PS', 'PB'],
        colors_5
    )
    
    plt.suptitle('Fuzzy Logic Membership Functions (v2.1)', fontsize=18, y=0.98)
    
    # =========================================================================
    # 儲存圖片
    # =========================================================================
    save_dir = Path(__file__).parent
    
    # 儲存整合圖
    combined_path = save_dir / 'fuzzy_membership_functions_v2.1.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"已儲存整合圖: {combined_path}")
    
    # 顯示圖片
    plt.show()