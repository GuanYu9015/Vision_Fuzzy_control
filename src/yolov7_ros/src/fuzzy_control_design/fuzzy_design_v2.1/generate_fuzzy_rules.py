#!/usr/bin/python3
"""
根據 fuzzy_rules_design_template.md 中的規則設計
產生完整的 625 條模糊規則 CSV 檔案
"""
import csv
import os
from itertools import product

# =============================================================================
# 模糊集合定義
# =============================================================================

# 輸入集合
E_D_SETS = ['VN', 'N', 'M', 'F', 'VF']       # Forward Distance Error
E_D_DOT_SETS = ['NB', 'NS', 'ZO', 'PS', 'PB']  # Forward Distance Error Rate
E_L_SETS = ['NB', 'NS', 'ZO', 'PS', 'PB']      # Lateral Error
E_L_DOT_SETS = ['NB', 'NS', 'ZO', 'PS', 'PB']  # Lateral Error Rate

# 輸出集合
V_SETS = ['S', 'VS', 'SL', 'M', 'F']           # Linear Velocity
OMEGA_SETS = ['NB', 'NS', 'ZO', 'PS', 'PB']    # Angular Velocity

# =============================================================================
# 表 1A：線速度基準規則 (只考慮 e_d 和 e_d_dot)
# 根據 fuzzy_rules_design_template.md 更新 (2025-12-26)
# =============================================================================
# 行: e_d (VN, N, M, F, VF)
# 列: e_d_dot (NB, NS, ZO, PS, PB)
V_BASE_RULES = {
    # e_d=VN (很近): 快速接近時停止，慢速遠離時可微動
    ('VN', 'NB'): 'S',   ('VN', 'NS'): 'S',   ('VN', 'ZO'): 'S',   ('VN', 'PS'): 'VS',  ('VN', 'PB'): 'VS',
    # e_d=N (近): 接近時停止或極慢，遠離時可慢速
    ('N',  'NB'): 'S',   ('N',  'NS'): 'VS',  ('N',  'ZO'): 'VS',  ('N',  'PS'): 'SL',  ('N',  'PB'): 'SL',
    # e_d=M (中): 穩定慢速，可根據趨勢加速
    ('M',  'NB'): 'SL',  ('M',  'NS'): 'SL',  ('M',  'ZO'): 'SL',  ('M',  'PS'): 'M',   ('M',  'PB'): 'F',
    # e_d=F (遠): 可較快前進
    ('F',  'NB'): 'SL',  ('F',  'NS'): 'M',   ('F',  'ZO'): 'M',   ('F',  'PS'): 'F',   ('F',  'PB'): 'F',
    # e_d=VF (很遠): 可高速前進
    ('VF', 'NB'): 'M',   ('VF', 'NS'): 'M',   ('VF', 'ZO'): 'F',   ('VF', 'PS'): 'F',   ('VF', 'PB'): 'F',
}

# =============================================================================
# 表 1B：角速度基準規則 (只考慮 e_l 和 e_l_dot)
# 根據 fuzzy_rules_design_template.md 更新 (2025-12-26)
# e_l: NB=左偏大, NS=左偏小, ZO=置中, PS=右偏小, PB=右偏大
# omega: PB=左轉大, PS=左轉小, ZO=直行, NS=右轉小, NB=右轉大
# 左偏需要右轉(正omega)來修正
# =============================================================================
# 行: e_l (NB, NS, ZO, PS, PB)
# 列: e_l_dot (NB, NS, ZO, PS, PB)
OMEGA_BASE_RULES = {
    # e_l=NB (左偏大): 需要強力右轉(PB)，除非已經在快速右移(PB)
    ('NB', 'NB'): 'PB',  ('NB', 'NS'): 'PB',  ('NB', 'ZO'): 'PB',  ('NB', 'PS'): 'PS',  ('NB', 'PB'): 'PS',
    # e_l=NS (左偏小): 需要右轉，但較溫和
    ('NS', 'NB'): 'PB',  ('NS', 'NS'): 'PB',  ('NS', 'ZO'): 'PS',  ('NS', 'PS'): 'PS',  ('NS', 'PB'): 'PS',
    # e_l=ZO (置中): 根據變化趨勢微調
    ('ZO', 'NB'): 'PS',  ('ZO', 'NS'): 'PS',  ('ZO', 'ZO'): 'ZO',  ('ZO', 'PS'): 'NS',  ('ZO', 'PB'): 'NS',
    # e_l=PS (右偏小): 需要左轉
    ('PS', 'NB'): 'NS',  ('PS', 'NS'): 'NS',  ('PS', 'ZO'): 'NS',  ('PS', 'PS'): 'NB',  ('PS', 'PB'): 'NB',
    # e_l=PB (右偏大): 需要強力左轉(NB)
    ('PB', 'NB'): 'NS',  ('PB', 'NS'): 'NS',  ('PB', 'ZO'): 'NB',  ('PB', 'PS'): 'NB',  ('PB', 'PB'): 'NB',
}

# =============================================================================
# 表 2A：線速度修正 (受橫向誤差 e_l 影響)
# 根據新設計: NB/PB 降二級，NS/PS 降一級
# =============================================================================
V_MODIFY_BY_E_L = {
    'NB': -2,  # 左偏大: 降二級，最低 S
    'NS': -1,  # 左偏小: 降一級
    'ZO': 0,   # 置中: 不修正
    'PS': -1,  # 右偏小: 降一級
    'PB': -2,  # 右偏大: 降二級，最低 S
}

# =============================================================================
# 表 2B：角速度修正 (受前方距離 e_d 影響)
# 根據純模糊設計: VN 減一級（低速少轉），N/M/F/VF 不修正
# =============================================================================
OMEGA_MODIFY_BY_E_D = {
    'VN': -1,  # 很近: 減一級（低速時減少角速度避免擺動）
    'N':  0,   # 近: 不修正
    'M':  0,   # 中: 不修正
    'F':  0,   # 遠: 不修正
    'VF': 0,   # 很遠: 不修正
}


def apply_v_level_change(base_v: str, delta: int) -> str:
    """
    對線速度應用等級變更
    降為負數，升為正數，但最低 S，最高 F
    """
    idx = V_SETS.index(base_v)
    new_idx = max(0, min(len(V_SETS) - 1, idx + delta))
    return V_SETS[new_idx]


def apply_omega_level_change(base_omega: str, delta: int) -> str:
    """
    對角速度應用等級變更
    delta 正數表示增強（往兩端靠攏），負數表示減弱（往 ZO 靠攏）
    
    注意：角速度的「加強」是遠離 ZO，「減弱」是靠近 ZO
    """
    if base_omega == 'ZO':
        return 'ZO'  # 本來就是零，無法加減
    
    idx = OMEGA_SETS.index(base_omega)
    center_idx = OMEGA_SETS.index('ZO')  # = 2
    
    # 計算到中心的距離
    distance_from_center = idx - center_idx  # 負表示左側(NB/NS)，正表示右側(PS/PB)
    
    if distance_from_center > 0:  # 正側 (PS, PB)
        # 加強 = 增加距離，減弱 = 減少距離
        new_distance = distance_from_center + delta
        new_distance = max(0, min(2, new_distance))  # 限制在 0~2
        new_idx = center_idx + new_distance
    else:  # 負側 (NB, NS)
        # 加強 = 增加距離(更負)，減弱 = 減少距離(靠近ZO)
        new_distance = -distance_from_center + delta
        new_distance = max(0, min(2, new_distance))  # 限制在 0~2
        new_idx = center_idx - new_distance
    
    new_idx = max(0, min(len(OMEGA_SETS) - 1, int(new_idx)))
    return OMEGA_SETS[new_idx]


def compute_rule(e_d: str, e_d_dot: str, e_l: str, e_l_dot: str) -> tuple:
    """
    根據四個輸入計算線速度和角速度輸出
    """
    # 1. 取得基準線速度 (根據 e_d 和 e_d_dot)
    base_v = V_BASE_RULES[(e_d, e_d_dot)]
    
    # 2. 應用線速度修正 (根據 e_l)
    v_delta = V_MODIFY_BY_E_L[e_l]
    final_v = apply_v_level_change(base_v, v_delta)
    
    # 3. 取得基準角速度 (根據 e_l 和 e_l_dot)
    base_omega = OMEGA_BASE_RULES[(e_l, e_l_dot)]
    
    # 4. 應用角速度修正 (根據 e_d)
    omega_delta = OMEGA_MODIFY_BY_E_D[e_d]
    final_omega = apply_omega_level_change(base_omega, omega_delta)
    
    return final_v, final_omega


def generate_complete_rules(output_path: str):
    """
    產生完整的 625 條模糊規則 CSV 檔案
    """
    headers = [
        'Rule_ID',
        'e_d (Forward Distance Error)',
        'e_d_dot (Forward Distance Error Rate)',
        'e_l (Lateral Error)',
        'e_l_dot (Lateral Error Rate)',
        'v (Linear Velocity)',
        'omega (Angular Velocity)',
    ]
    
    all_combinations = list(product(E_D_SETS, E_D_DOT_SETS, E_L_SETS, E_L_DOT_SETS))
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for rule_id, (e_d, e_d_dot, e_l, e_l_dot) in enumerate(all_combinations, start=1):
            v_output, omega_output = compute_rule(e_d, e_d_dot, e_l, e_l_dot)
            
            writer.writerow([
                rule_id,
                e_d,
                e_d_dot,
                e_l,
                e_l_dot,
                v_output,
                omega_output,
            ])
    
    print(f"已產生 {len(all_combinations)} 條模糊規則")
    print(f"CSV 檔案已儲存至: {output_path}")
    
    # 顯示統計資訊
    print("\n=== 規則統計 ===")
    v_counts = {v: 0 for v in V_SETS}
    omega_counts = {o: 0 for o in OMEGA_SETS}
    
    for e_d, e_d_dot, e_l, e_l_dot in all_combinations:
        v, omega = compute_rule(e_d, e_d_dot, e_l, e_l_dot)
        v_counts[v] += 1
        omega_counts[omega] += 1
    
    print("\n線速度分佈:")
    for v in V_SETS:
        print(f"  {v}: {v_counts[v]} 條 ({v_counts[v]/625*100:.1f}%)")
    
    print("\n角速度分佈:")
    for o in OMEGA_SETS:
        print(f"  {o}: {omega_counts[o]} 條 ({omega_counts[o]/625*100:.1f}%)")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'fuzzy_rules_relaxed.csv')
    generate_complete_rules(output_file)
