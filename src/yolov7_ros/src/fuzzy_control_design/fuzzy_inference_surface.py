#!/usr/bin/python3
"""
模糊推論曲面繪製程式
Fuzzy Inference Surface Visualization

根據 mod_fuzzy_control4.py 的設計繪製控制曲面
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
from typing import Tuple, Dict

# =============================================================================
# 隸屬函數定義 (與 mod_fuzzy_control4.py 一致)
# =============================================================================

class MembershipFunctions:
    """隸屬函數定義"""
    
    # 前方距離誤差 e_d [0, 2.08]
    E_D = {
        'VN': (0.0, 0.0, 0.52),       # Very Near
        'N': (0.0, 0.52, 1.04),       # Near
        'M': (0.52, 1.04, 1.56),      # Medium
        'F': (1.04, 1.56, 2.08),      # Far
        'VF': (1.56, 2.08, 2.08)      # Very Far
    }
    
    # 前方距離變化率 e_d_dot [-2.08, 2.08]
    E_D_DOT = {
        'NB': (-2.08, -2.08, -1.04),  # Negative Big
        'NS': (-2.08, -1.04, 0.0),    # Negative Small
        'ZO': (-1.04, 0.0, 1.04),     # Zero
        'PS': (0.0, 1.04, 2.08),      # Positive Small
        'PB': (1.04, 2.08, 2.08)      # Positive Big
    }
    
    # 橫向誤差 e_l [-0.5, 0.5] 非對稱死區設計
    E_L = {
        'NB': (-1.0, -0.5, -0.15),     # Negative Big
        'NS': (-0.5, -0.15, -0.05),    # Negative Small
        'ZO': (-0.08, 0.0, 0.08),      # Zero
        'PS': (0.05, 0.15, 0.5),       # Positive Small
        'PB': (0.15, 0.5, 1.0)         # Positive Big
    }
    
    # 橫向變化率 e_l_dot [-1.0, 1.0]
    E_L_DOT = {
        'NB': (-1.5, -1.0, -0.5),
        'NS': (-1.0, -0.5, 0.0),
        'ZO': (-0.5, 0.0, 0.5),
        'PS': (0.0, 0.5, 1.0),
        'PB': (0.5, 1.0, 1.0)
    }
    
    # 線速度輸出 v (Singleton)
    V = {
        'S': 0.0,
        'VS': 0.18,
        'SL': 0.32,
        'M': 0.45,
        'F': 0.55
    }
    
    # 角速度輸出 omega (Singleton)
    OMEGA = {
        'NB': -1.4,
        'NS': -0.5,
        'ZO': 0.0,
        'PS': 0.5,
        'PB': 1.4
    }


def triangular_mf(x: float, params: Tuple[float, float, float]) -> float:
    """三角形隸屬函數"""
    a, b, c = params
    
    if x <= a:
        return 1.0 if a == b else 0.0
    elif x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    elif x <= c:
        return (c - x) / (c - b) if c > b else 1.0
    else:
        return 1.0 if b == c else 0.0


def get_membership(value: float, mf_dict: dict) -> Dict[str, float]:
    """計算輸入值在所有隸屬函數的隸屬度"""
    memberships = {}
    for label, params in mf_dict.items():
        memberships[label] = triangular_mf(value, params)
    return memberships


# =============================================================================
# 模糊規則 (從 generate_fuzzy_rules.py 載入邏輯)
# =============================================================================

# 表 1A：線速度基準規則
V_BASE_RULES = {
    ('VN', 'NB'): 'S',   ('VN', 'NS'): 'S',   ('VN', 'ZO'): 'S',   ('VN', 'PS'): 'VS',  ('VN', 'PB'): 'VS',
    ('N',  'NB'): 'S',   ('N',  'NS'): 'VS',  ('N',  'ZO'): 'VS',  ('N',  'PS'): 'SL',  ('N',  'PB'): 'SL',
    ('M',  'NB'): 'SL',  ('M',  'NS'): 'SL',  ('M',  'ZO'): 'SL',  ('M',  'PS'): 'M',   ('M',  'PB'): 'F',
    ('F',  'NB'): 'SL',  ('F',  'NS'): 'M',   ('F',  'ZO'): 'M',   ('F',  'PS'): 'F',   ('F',  'PB'): 'F',
    ('VF', 'NB'): 'M',   ('VF', 'NS'): 'M',   ('VF', 'ZO'): 'F',   ('VF', 'PS'): 'F',   ('VF', 'PB'): 'F',
}

# 表 1B：角速度基準規則
OMEGA_BASE_RULES = {
    ('NB', 'NB'): 'PB',  ('NB', 'NS'): 'PB',  ('NB', 'ZO'): 'PB',  ('NB', 'PS'): 'PS',  ('NB', 'PB'): 'PS',
    ('NS', 'NB'): 'PB',  ('NS', 'NS'): 'PB',  ('NS', 'ZO'): 'PS',  ('NS', 'PS'): 'PS',  ('NS', 'PB'): 'PS',
    ('ZO', 'NB'): 'PS',  ('ZO', 'NS'): 'PS',  ('ZO', 'ZO'): 'ZO',  ('ZO', 'PS'): 'NS',  ('ZO', 'PB'): 'NS',
    ('PS', 'NB'): 'NS',  ('PS', 'NS'): 'NS',  ('PS', 'ZO'): 'NS',  ('PS', 'PS'): 'NB',  ('PS', 'PB'): 'NB',
    ('PB', 'NB'): 'NS',  ('PB', 'NS'): 'NS',  ('PB', 'ZO'): 'NB',  ('PB', 'PS'): 'NB',  ('PB', 'PB'): 'NB',
}

# 表 2A：線速度修正
V_MODIFY_BY_E_L = {'NB': -2, 'NS': -1, 'ZO': 0, 'PS': -1, 'PB': -2}

# 表 2B：角速度修正
OMEGA_MODIFY_BY_E_D = {'VN': -1, 'N': 0, 'M': 0, 'F': 0, 'VF': 0}

V_SETS = ['S', 'VS', 'SL', 'M', 'F']
OMEGA_SETS = ['NB', 'NS', 'ZO', 'PS', 'PB']


def apply_v_level_change(base_v: str, delta: int) -> str:
    """對線速度應用等級變更"""
    idx = V_SETS.index(base_v)
    new_idx = max(0, min(len(V_SETS) - 1, idx + delta))
    return V_SETS[new_idx]


def apply_omega_level_change(base_omega: str, delta: int) -> str:
    """對角速度應用等級變更"""
    if base_omega == 'ZO':
        return 'ZO'
    
    idx = OMEGA_SETS.index(base_omega)
    center_idx = OMEGA_SETS.index('ZO')
    distance_from_center = idx - center_idx
    
    if distance_from_center > 0:
        new_distance = distance_from_center + delta
        new_distance = max(0, min(2, new_distance))
        new_idx = center_idx + new_distance
    else:
        new_distance = -distance_from_center + delta
        new_distance = max(0, min(2, new_distance))
        new_idx = center_idx - new_distance
    
    new_idx = max(0, min(len(OMEGA_SETS) - 1, int(new_idx)))
    return OMEGA_SETS[new_idx]


def get_rule_output(e_d_label: str, e_d_dot_label: str, 
                    e_l_label: str, e_l_dot_label: str) -> Tuple[str, str]:
    """取得規則的輸出標籤"""
    base_v = V_BASE_RULES[(e_d_label, e_d_dot_label)]
    v_delta = V_MODIFY_BY_E_L[e_l_label]
    final_v = apply_v_level_change(base_v, v_delta)
    
    base_omega = OMEGA_BASE_RULES[(e_l_label, e_l_dot_label)]
    omega_delta = OMEGA_MODIFY_BY_E_D[e_d_label]
    final_omega = apply_omega_level_change(base_omega, omega_delta)
    
    return final_v, final_omega


# =============================================================================
# 模糊推論引擎
# =============================================================================

def fuzzy_inference(e_d: float, e_d_dot: float, e_l: float, e_l_dot: float) -> Tuple[float, float]:
    """
    執行模糊推論
    
    Returns:
        (v, omega): 線速度與角速度
    """
    mf = MembershipFunctions()
    
    # 模糊化
    mu_e_d = get_membership(e_d, mf.E_D)
    mu_e_d_dot = get_membership(e_d_dot, mf.E_D_DOT)
    mu_e_l = get_membership(e_l, mf.E_L)
    mu_e_l_dot = get_membership(e_l_dot, mf.E_L_DOT)
    
    # 規則評估
    v_numerator = 0.0
    v_denominator = 0.0
    omega_numerator = 0.0
    omega_denominator = 0.0
    
    for e_d_label in mf.E_D.keys():
        for e_d_dot_label in mf.E_D_DOT.keys():
            for e_l_label in mf.E_L.keys():
                for e_l_dot_label in mf.E_L_DOT.keys():
                    # 計算激發強度
                    firing_strength = min(
                        mu_e_d.get(e_d_label, 0),
                        mu_e_d_dot.get(e_d_dot_label, 0),
                        mu_e_l.get(e_l_label, 0),
                        mu_e_l_dot.get(e_l_dot_label, 0)
                    )
                    
                    if firing_strength > 0:
                        v_label, omega_label = get_rule_output(
                            e_d_label, e_d_dot_label, e_l_label, e_l_dot_label
                        )
                        
                        v_output = mf.V[v_label]
                        omega_output = mf.OMEGA[omega_label]
                        
                        v_numerator += firing_strength * v_output
                        v_denominator += firing_strength
                        omega_numerator += firing_strength * omega_output
                        omega_denominator += firing_strength
    
    # 解模糊
    v = v_numerator / v_denominator if v_denominator > 0 else 0.0
    omega = omega_numerator / omega_denominator if omega_denominator > 0 else 0.0
    
    return v, omega


# =============================================================================
# 繪製控制曲面
# =============================================================================

def plot_v_surface_ed_el(resolution: int = 50, save_path: str = None):
    """
    繪製線速度曲面: v = f(e_d, e_l)
    固定 e_d_dot=0, e_l_dot=0
    """
    e_d_range = np.linspace(0, 2.08, resolution)
    e_l_range = np.linspace(-0.5, 0.5, resolution)
    
    E_D, E_L = np.meshgrid(e_d_range, e_l_range)
    V = np.zeros_like(E_D)
    
    for i in range(resolution):
        for j in range(resolution):
            v, _ = fuzzy_inference(E_D[i, j], 0.0, E_L[i, j], 0.0)
            V[i, j] = v
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(E_D, E_L, V, cmap=cm.viridis, 
                           linewidth=0, antialiased=True, alpha=0.9)
    
    ax.set_xlabel('e_d (Forward Distance Error) [m]', fontsize=12)
    ax.set_ylabel('e_l (Lateral Error) [m]', fontsize=12)
    ax.set_zlabel('v (Linear Velocity) [m/s]', fontsize=12)
    ax.set_title('Fuzzy Inference Surface: Linear Velocity\n(e_d_dot=0, e_l_dot=0)', fontsize=14)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='v [m/s]')
    
    ax.view_init(elev=30, azim=225)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已儲存: {save_path}")
    
    return fig


def plot_omega_surface_el_eldot(resolution: int = 50, save_path: str = None):
    """
    繪製角速度曲面: omega = f(e_l, e_l_dot)
    固定 e_d=1.04 (停止距離), e_d_dot=0
    """
    e_l_range = np.linspace(-0.5, 0.5, resolution)
    e_l_dot_range = np.linspace(-1.0, 1.0, resolution)
    
    E_L, E_L_DOT = np.meshgrid(e_l_range, e_l_dot_range)
    OMEGA = np.zeros_like(E_L)
    
    for i in range(resolution):
        for j in range(resolution):
            _, omega = fuzzy_inference(1.04, 0.0, E_L[i, j], E_L_DOT[i, j])
            OMEGA[i, j] = omega
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(E_L, E_L_DOT, OMEGA, cmap=cm.coolwarm, 
                           linewidth=0, antialiased=True, alpha=0.9)
    
    ax.set_xlabel('e_l (Lateral Error) [m]', fontsize=12)
    ax.set_ylabel('e_l_dot (Lateral Error Rate) [m/s]', fontsize=12)
    ax.set_zlabel('ω (Angular Velocity) [rad/s]', fontsize=12)
    ax.set_title('Fuzzy Inference Surface: Angular Velocity\n(e_d=1.04m, e_d_dot=0)', fontsize=14)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='ω [rad/s]')
    
    ax.view_init(elev=30, azim=225)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已儲存: {save_path}")
    
    return fig


def plot_v_surface_ed_eddot(resolution: int = 50, save_path: str = None):
    """
    繪製線速度曲面: v = f(e_d, e_d_dot)
    固定 e_l=0 (置中), e_l_dot=0
    """
    e_d_range = np.linspace(0, 2.08, resolution)
    e_d_dot_range = np.linspace(-2.08, 2.08, resolution)
    
    E_D, E_D_DOT = np.meshgrid(e_d_range, e_d_dot_range)
    V = np.zeros_like(E_D)
    
    for i in range(resolution):
        for j in range(resolution):
            v, _ = fuzzy_inference(E_D[i, j], E_D_DOT[i, j], 0.0, 0.0)
            V[i, j] = v
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(E_D, E_D_DOT, V, cmap=cm.plasma, 
                           linewidth=0, antialiased=True, alpha=0.9)
    
    ax.set_xlabel('e_d (Forward Distance Error) [m]', fontsize=12)
    ax.set_ylabel('e_d_dot (Distance Rate) [m/s]', fontsize=12)
    ax.set_zlabel('v (Linear Velocity) [m/s]', fontsize=12)
    ax.set_title('Fuzzy Inference Surface: Linear Velocity\n(e_l=0, e_l_dot=0)', fontsize=14)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='v [m/s]')
    
    ax.view_init(elev=30, azim=225)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已儲存: {save_path}")
    
    return fig


def plot_omega_surface_el_ed(resolution: int = 50, save_path: str = None):
    """
    繪製角速度曲面: omega = f(e_l, e_d)
    顯示距離如何影響角速度
    固定 e_d_dot=0, e_l_dot=0
    """
    e_l_range = np.linspace(-0.5, 0.5, resolution)
    e_d_range = np.linspace(0, 2.08, resolution)
    
    E_L, E_D = np.meshgrid(e_l_range, e_d_range)
    OMEGA = np.zeros_like(E_L)
    
    for i in range(resolution):
        for j in range(resolution):
            _, omega = fuzzy_inference(E_D[i, j], 0.0, E_L[i, j], 0.0)
            OMEGA[i, j] = omega
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(E_L, E_D, OMEGA, cmap=cm.coolwarm, 
                           linewidth=0, antialiased=True, alpha=0.9)
    
    ax.set_xlabel('e_l (Lateral Error) [m]', fontsize=12)
    ax.set_ylabel('e_d (Forward Distance Error) [m]', fontsize=12)
    ax.set_zlabel('ω (Angular Velocity) [rad/s]', fontsize=12)
    ax.set_title('Fuzzy Inference Surface: Angular Velocity\n(e_d_dot=0, e_l_dot=0)', fontsize=14)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='ω [rad/s]')
    
    ax.view_init(elev=30, azim=225)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已儲存: {save_path}")
    
    return fig


def plot_all_surfaces(output_dir: str = None):
    """繪製所有控制曲面"""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在繪製模糊推論曲面...")
    print("=" * 50)
    
    # 1. 線速度曲面 (e_d vs e_l)
    print("\n[1/4] 繪製 v = f(e_d, e_l)...")
    plot_v_surface_ed_el(
        resolution=60,
        save_path=os.path.join(output_dir, 'surface_v_ed_el.png')
    )
    
    # 2. 線速度曲面 (e_d vs e_d_dot)
    print("[2/4] 繪製 v = f(e_d, e_d_dot)...")
    plot_v_surface_ed_eddot(
        resolution=60,
        save_path=os.path.join(output_dir, 'surface_v_ed_eddot.png')
    )
    
    # 3. 角速度曲面 (e_l vs e_l_dot)
    print("[3/4] 繪製 ω = f(e_l, e_l_dot)...")
    plot_omega_surface_el_eldot(
        resolution=60,
        save_path=os.path.join(output_dir, 'surface_omega_el_eldot.png')
    )
    
    # 4. 角速度曲面 (e_l vs e_d)
    print("[4/4] 繪製 ω = f(e_l, e_d)...")
    plot_omega_surface_el_ed(
        resolution=60,
        save_path=os.path.join(output_dir, 'surface_omega_el_ed.png')
    )
    
    print("\n" + "=" * 50)
    print(f"所有曲面已儲存至: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    plot_all_surfaces()
    plt.show()
