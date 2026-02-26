#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模糊規則驗證與統計腳本
驗證 fuzzy_rules_simplified.csv 的完整性並產生統計報告
"""

import csv
from pathlib import Path
from collections import Counter

def validate_and_analyze():
    """驗證規則檔案並產生統計分析"""
    
    script_dir = Path(__file__).parent
    rules_path = script_dir / 'fuzzy_rules_relaxed.csv'
    
    print("=" * 60)
    print("模糊規則驗證報告")
    print("=" * 60)
    
    # 定義有效的模糊集合
    valid_sets = {
        'e_d': ['VN', 'N', 'M', 'F', 'VF'],
        'e_d_dot': ['NB', 'NS', 'ZO', 'PS', 'PB'],
        'e_l': ['NB', 'NS', 'ZO', 'PS', 'PB'],
        'e_l_dot': ['NB', 'NS', 'ZO', 'PS', 'PB'],
        'v': ['S', 'VS', 'SL', 'M', 'F'],
        'omega': ['NB', 'NS', 'ZO', 'PS', 'PB']
    }
    
    # 讀取規則
    rules = []
    errors = []
    
    try:
        with open(rules_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rules.append(row)
    except FileNotFoundError:
        print(f"[錯誤] 找不到規則檔案: {rules_path}")
        return False
    except Exception as e:
        print(f"[錯誤] 讀取檔案失敗: {e}")
        return False
    
    print(f"\n📁 規則檔案: {rules_path}")
    print(f"📊 規則總數: {len(rules)} 條")
    
    # 預期規則數量
    expected_count = 5 * 5 * 5 * 5  # 625
    if len(rules) != expected_count:
        errors.append(f"規則數量不正確: 預期 {expected_count}，實際 {len(rules)}")
    
    # 驗證每條規則
    v_counter = Counter()
    omega_counter = Counter()
    
    for i, rule in enumerate(rules, 1):
        rule_id = rule.get('Rule_ID', str(i))
        
        # 驗證輸入變數
        e_d = rule.get('e_d (Forward Distance Error)', '')
        e_d_dot = rule.get('e_d_dot (Forward Distance Error Rate)', '')
        e_l = rule.get('e_l (Lateral Error)', '')
        e_l_dot = rule.get('e_l_dot (Lateral Error Rate)', '')
        v = rule.get('v (Linear Velocity)', '')
        omega = rule.get('omega (Angular Velocity)', '')
        
        if e_d not in valid_sets['e_d']:
            errors.append(f"Rule {rule_id}: 無效的 e_d 值 '{e_d}'")
        if e_d_dot not in valid_sets['e_d_dot']:
            errors.append(f"Rule {rule_id}: 無效的 e_d_dot 值 '{e_d_dot}'")
        if e_l not in valid_sets['e_l']:
            errors.append(f"Rule {rule_id}: 無效的 e_l 值 '{e_l}'")
        if e_l_dot not in valid_sets['e_l_dot']:
            errors.append(f"Rule {rule_id}: 無效的 e_l_dot 值 '{e_l_dot}'")
        if v not in valid_sets['v']:
            errors.append(f"Rule {rule_id}: 無效的 v 值 '{v}'")
        if omega not in valid_sets['omega']:
            errors.append(f"Rule {rule_id}: 無效的 omega 值 '{omega}'")
        
        # 統計輸出分布
        v_counter[v] += 1
        omega_counter[omega] += 1
    
    # 檢查規則完整性（所有組合都有涵蓋）
    seen_combinations = set()
    for rule in rules:
        e_d = rule.get('e_d (Forward Distance Error)', '')
        e_d_dot = rule.get('e_d_dot (Forward Distance Error Rate)', '')
        e_l = rule.get('e_l (Lateral Error)', '')
        e_l_dot = rule.get('e_l_dot (Lateral Error Rate)', '')
        combo = (e_d, e_d_dot, e_l, e_l_dot)
        if combo in seen_combinations:
            errors.append(f"發現重複規則: {combo}")
        seen_combinations.add(combo)
    
    # 檢查是否有缺失的組合
    expected_combos = set()
    for e_d in valid_sets['e_d']:
        for e_d_dot in valid_sets['e_d_dot']:
            for e_l in valid_sets['e_l']:
                for e_l_dot in valid_sets['e_l_dot']:
                    expected_combos.add((e_d, e_d_dot, e_l, e_l_dot))
    
    missing = expected_combos - seen_combinations
    if missing:
        errors.append(f"缺少 {len(missing)} 個規則組合")
        for combo in list(missing)[:5]:
            errors.append(f"  缺少: {combo}")
    
    # 輸出驗證結果
    print("\n" + "-" * 60)
    print("✅ 驗證結果")
    print("-" * 60)
    
    if errors:
        print(f"\n❌ 發現 {len(errors)} 個錯誤:")
        for err in errors[:10]:
            print(f"  • {err}")
        if len(errors) > 10:
            print(f"  ... 還有 {len(errors) - 10} 個錯誤")
        validation_passed = False
    else:
        print("\n✓ 所有規則驗證通過！")
        print("✓ 規則數量正確 (625 條)")
        print("✓ 所有輸入/輸出值有效")
        print("✓ 所有組合完整覆蓋")
        print("✓ 無重複規則")
        validation_passed = True
    
    # 輸出統計分析
    print("\n" + "-" * 60)
    print("📈 輸出值分布統計")
    print("-" * 60)
    
    print("\n【線速度 v 分布】")
    print(f"{'值':<8} {'數量':<8} {'比例':<10} {'視覺化'}")
    print("-" * 50)
    v_order = ['S', 'VS', 'SL', 'M', 'F']
    v_labels = {'S': '停止', 'VS': '極慢', 'SL': '慢速', 'M': '中速', 'F': '快速'}
    for v in v_order:
        count = v_counter[v]
        pct = count / len(rules) * 100
        bar = '█' * int(pct / 2)
        print(f"{v:<4} ({v_labels[v]:<4}) {count:<8} {pct:>5.1f}%    {bar}")
    
    print("\n【角速度 ω 分布】")
    print(f"{'值':<8} {'數量':<8} {'比例':<10} {'視覺化'}")
    print("-" * 50)
    omega_order = ['NB', 'NS', 'ZO', 'PS', 'PB']
    omega_labels = {'NB': '大右轉', 'NS': '小右轉', 'ZO': '直行', 'PS': '小左轉', 'PB': '大左轉'}
    for o in omega_order:
        count = omega_counter[o]
        pct = count / len(rules) * 100
        bar = '█' * int(pct / 2)
        print(f"{o:<4} ({omega_labels[o]:<4}) {count:<8} {pct:>5.1f}%    {bar}")
    
    # 對稱性分析
    print("\n【對稱性分析】")
    right_omega = omega_counter['NB'] + omega_counter['NS']  # 負值 = 右轉
    left_omega = omega_counter['PS'] + omega_counter['PB']   # 正值 = 左轉
    print(f"左轉規則 (PS+PB): {left_omega} 條")
    print(f"右轉規則 (NB+NS): {right_omega} 條")
    print(f"直行規則 (ZO):    {omega_counter['ZO']} 條")
    
    if left_omega == right_omega:
        print("✓ 左右轉向規則對稱")
    else:
        print(f"⚠ 左右轉向規則不對稱 (差異: {abs(left_omega - right_omega)} 條)")
    
    print("\n" + "=" * 60)
    print("驗證完成")
    print("=" * 60)
    
    return validation_passed

if __name__ == '__main__':
    validate_and_analyze()
