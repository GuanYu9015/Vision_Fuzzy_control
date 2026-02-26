#!/usr/bin/python3
"""
四輸入模糊控制器
4-Input Fuzzy Controller for Obstacle Avoidance

輸入變數 (4個):
- e_d: 前方距離誤差 [0, 2.08] m
- e_d_dot: 前方距離變化率 [-2.08, 2.08]
- e_l: 橫向誤差 [-0.5, 0.5] m
- e_l_dot: 橫向變化率 [-1.0, 1.0]

輸出變數 (2個):
- v: 線速度 {0, 0.225, 0.45, 0.675, 0.9} m/s
- omega: 角速度 {-2.5, -1.25, 0, 1.25, 2.5} rad/s

規則數量: 625 條 (5^4)
"""

import os
import csv
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

import rospy
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist


class MembershipFunctions:
    """隸屬函數定義"""
    
    # 前方距離誤差 e_d [0, 2.08]
    E_D = {
        'VN': (0.0, 0.0, 0.52),       # Very Near: 梯形左半
        'N': (0.0, 0.52, 1.04),       # Near
        'M': (0.52, 1.04, 1.56),      # Medium
        'F': (1.04, 1.56, 2.08),      # Far
        'VF': (1.56, 2.08, 2.08)      # Very Far: 梯形右半
    }
    
    # 前方距離變化率 e_d_dot [-2.08, 2.08]
    E_D_DOT = {
        'NB': (-2.08, -2.08, -1.04),  # Negative Big
        'NS': (-2.08, -1.04, 0.0),    # Negative Small
        'ZO': (-1.04, 0.0, 1.04),     # Zero
        'PS': (0.0, 1.04, 2.08),      # Positive Small
        'PB': (1.04, 2.08, 2.08)      # Positive Big
    }
    
    # 橫向誤差 e_l [-0.5, 0.5] - 擴大死區減少穩態抖動
    E_L = {
        'NB': (-1.0, -0.5, -0.15),     # Negative Big (左偏大)
        'NS': (-0.5, -0.15, -0.05),    # Negative Small (終點內縮，創造死區)
        'ZO': (-0.08, 0.0, 0.08),      # Zero (擴大死區至 ±0.08m)
        'PS': (0.05, 0.15, 0.5),       # Positive Small (起點外推，創造死區)
        'PB': (0.15, 0.5, 1.0)         # Positive Big (右偏大)
    }
    
    # 橫向變化率 e_l_dot [-1.0, 1.0]
    E_L_DOT = {
        'NB': (-1.5, -1.0, -0.5),     # Negative Big
        'NS': (-1.0, -0.5, 0.0),      # Negative Small
        'ZO': (-0.5, 0.0, 0.5),       # Zero
        'PS': (0.0, 0.5, 1.0),        # Positive Small
        'PB': (0.5, 1.0, 1.5)         # Positive Big
    }
    
    # 線速度輸出 v (Singleton) - 壓縮低速區間讓動作更平滑
    V = {
        'S': 0.0,       # Stop
        'VS': 0.18,     # Very Slow (提高)
        'SL': 0.32,     # Slow (提高)
        'M': 0.45,      # Medium
        'F': 0.55       # Fast (降低避免高速衝刺)
    }
    
    # 角速度輸出 omega (Singleton) - 降低 NS/PS 減少擺動
    OMEGA = {
        'NB': -1.4,    # Negative Big (右轉大) - 降低
        'NS': -0.5,    # Negative Small (進一步降低)
        'ZO': 0.0,     # Zero
        'PS': 0.5,     # Positive Small (進一步降低)
        'PB': 1.4      # Positive Big (左轉大) - 降低
    }


def triangular_mf(x: float, params: Tuple[float, float, float]) -> float:
    """
    三角形/梯形隸屬函數
    
    Args:
        x: 輸入值
        params: (a, b, c) - 三角形的左、峰、右座標
        
    Returns:
        隸屬度 [0, 1]
    """
    a, b, c = params
    
    if x <= a:
        return 1.0 if a == b else 0.0
    elif x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    elif x <= c:
        return (c - x) / (c - b) if c > b else 1.0
    else:
        return 1.0 if b == c else 0.0


class FuzzyController4Input:
    """四輸入模糊控制器"""
    
    def __init__(self, rules_csv_path: Optional[str] = None):
        """
        初始化模糊控制器
        
        Args:
            rules_csv_path: 規則 CSV 檔案路徑
        """
        # 設定規則檔案路徑
        if rules_csv_path is None:
            script_dir = Path(__file__).parent.parent
            rules_csv_path = script_dir / 'src' / 'fuzzy_control_design' / 'fuzzy_rules_relaxed.csv'
        
        self.rules_csv_path = Path(rules_csv_path)
        
        # 載入規則
        self.rules = self._load_rules()
        
        # 隸屬函數
        self.mf = MembershipFunctions()
        
        # 低通濾波器參數 (v2.1: 恢復濾波以解決移動猶豫)
        # alpha_v=0.2: 強濾波，平滑速度變化
        # alpha_omega=0.4: 中度濾波，減少轉彎擺動
        self.alpha_v = 0.2
        self.alpha_omega = 0.3
        
        # 濾波器狀態
        self.prev_v = 0.0
        self.prev_omega = 0.0
        self.filter_initialized = False
        
        # 日誌控制
        self.log_interval = 1.0  # 秒
        self.last_log_time = 0
        
        print(f"模糊控制器已初始化，載入規則數量: {len(self.rules)} 條")
        print(f"低通濾波器: v={self.alpha_v}, omega={self.alpha_omega}")
    
    def _load_rules(self) -> list:
        """載入模糊規則"""
        if not self.rules_csv_path.exists():
            raise FileNotFoundError(f"找不到規則檔案: {self.rules_csv_path}")
        
        rules = []
        
        with open(self.rules_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rule = {
                    'e_d': row['e_d (Forward Distance Error)'].strip(),
                    'e_d_dot': row['e_d_dot (Forward Distance Error Rate)'].strip(),
                    'e_l': row['e_l (Lateral Error)'].strip(),
                    'e_l_dot': row['e_l_dot (Lateral Error Rate)'].strip(),
                    'v': row['v (Linear Velocity)'].strip(),
                    'omega': row['omega (Angular Velocity)'].strip()
                }
                rules.append(rule)
        
        return rules
    
    def _get_membership(self, value: float, mf_dict: dict) -> Dict[str, float]:
        """
        計算輸入值在所有隸屬函數的隸屬度
        
        Args:
            value: 輸入值
            mf_dict: 隸屬函數定義字典
            
        Returns:
            各語言變數的隸屬度
        """
        memberships = {}
        for label, params in mf_dict.items():
            memberships[label] = triangular_mf(value, params)
        return memberships
    
    def compute(self, 
                e_d: float, 
                e_d_dot: float, 
                e_l: float, 
                e_l_dot: float) -> Tuple[float, float]:
        """
        執行模糊推論
        
        Args:
            e_d: 前方距離誤差 [0, 2.08] m
            e_d_dot: 距離變化率 [-2.08, 2.08]
            e_l: 橫向誤差 [-0.5, 0.5] m
            e_l_dot: 橫向變化率 [-1.0, 1.0]
            
        Returns:
            (v, omega): 線速度 (m/s) 與角速度 (rad/s)
        """
        # 步驟 1: 模糊化 - 計算各輸入的隸屬度
        mu_e_d = self._get_membership(e_d, self.mf.E_D)
        mu_e_d_dot = self._get_membership(e_d_dot, self.mf.E_D_DOT)
        mu_e_l = self._get_membership(e_l, self.mf.E_L)
        mu_e_l_dot = self._get_membership(e_l_dot, self.mf.E_L_DOT)
        
        # 步驟 2: 規則評估與聚合
        v_numerator = 0.0
        v_denominator = 0.0
        omega_numerator = 0.0
        omega_denominator = 0.0
        
        for rule in self.rules:
            # 計算規則激發強度（AND 運算 = 取最小值）
            firing_strength = min(
                mu_e_d.get(rule['e_d'], 0),
                mu_e_d_dot.get(rule['e_d_dot'], 0),
                mu_e_l.get(rule['e_l'], 0),
                mu_e_l_dot.get(rule['e_l_dot'], 0)
            )
            
            if firing_strength > 0:
                # 取得輸出的 singleton 值
                v_output = self.mf.V.get(rule['v'], 0)
                omega_output = self.mf.OMEGA.get(rule['omega'], 0)
                
                # 加權平均聚合
                v_numerator += firing_strength * v_output
                v_denominator += firing_strength
                omega_numerator += firing_strength * omega_output
                omega_denominator += firing_strength
        
        # 步驟 3: 解模糊化（加權平均法）
        if v_denominator > 0:
            v = v_numerator / v_denominator
        else:
            v = 0.0  # 預設停止
        
        if omega_denominator > 0:
            omega = omega_numerator / omega_denominator
        else:
            omega = 0.0  # 預設直行
        
        # ========== 純模糊控制器：範圍限制 + 低通濾波 ==========
        
        # 1. 範圍限制 (安全網)
        v = np.clip(v, 0.0, 0.55)
        omega = np.clip(omega, -1.4, 1.4)
        
        # 2. 低通濾波 (解決移動猶豫與抖動)
        if not self.filter_initialized:
            self.prev_v = v
            self.prev_omega = omega
            self.filter_initialized = True
        else:
            v = self.alpha_v * v + (1 - self.alpha_v) * self.prev_v
            omega = self.alpha_omega * omega + (1 - self.alpha_omega) * self.prev_omega
            
            self.prev_v = v
            self.prev_omega = omega
            
        # =======================================================
        
        return float(v), float(omega)
    
    def compute_with_logging(self,
                             e_d: float,
                             e_d_dot: float,
                             e_l: float,
                             e_l_dot: float) -> Tuple[float, float]:
        """
        執行模糊推論並輸出日誌
        """
        v, omega = self.compute(e_d, e_d_dot, e_l, e_l_dot)
        
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            rospy.loginfo(f"Fuzzy: e_d={e_d:.3f}, e_l={e_l:.3f} -> v={v:.3f}, omega={omega:.3f}")
            self.last_log_time = current_time
        
        return v, omega


class FuzzyControllerNode:
    """ROS 模糊控制器節點"""
    
    def __init__(self):
        """初始化節點"""
        rospy.init_node('fuzzy_controller_4input_node', anonymous=True)
        
        # 載入模糊控制器
        rules_path = rospy.get_param('~rules_path', None)
        self.controller = FuzzyController4Input(rules_path)
        
        # 訂閱者
        rospy.Subscriber('/road_info', Float32MultiArray, self.road_info_callback)
        rospy.Subscriber('/avoidance_enabled', String, self.avoidance_control_callback)
        rospy.Subscriber('/system_status', String, self.system_status_callback)
        rospy.Subscriber('/voice_command_code', String, self.voice_command_callback)
        
        # 發布者
        self.cmd_vel_pub = rospy.Publisher('/fuzzy_cmd_vel', Twist, queue_size=10)
        
        # 狀態
        self.system_active = False
        self.task_started = False
        self.avoid_obstacle = False
        self.keyboard_enabled = False  # 鍵盤控制的避障開關
        
        # 模糊推論時間統計
        self.inference_times = []
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.log_interval = 5.0
        self.last_log_time = time.time()
        self.start_time = time.time()
        
        # 建立 log 檔案
        log_timestamp = time.strftime("%Y%m%d_%H%M%S")
        script_dir = Path(__file__).parent.parent
        log_dir = script_dir / 'outputs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"fuzzy_stats_{log_timestamp}.txt"
        
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(f"# Fuzzy Inference Stats - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Format: timestamp, avg_inference_ms, total_avg_ms, count\n")
            f.write("-" * 50 + "\n")
        
        rospy.loginfo(f"Fuzzy stats log: {self.log_path}")
        
        # 建立輸入輸出詳細 log (CSV 格式)
        self.io_log_path = log_dir / f"fuzzy_io_{log_timestamp}.csv"
        with open(self.io_log_path, 'w', encoding='utf-8') as f:
            f.write("timestamp,e_d,e_d_dot,e_l,e_l_dot,v,omega\n")
        
        rospy.loginfo(f"Fuzzy I/O log: {self.io_log_path}")
        
        # 執行頻率
        self.rate = rospy.Rate(10)  # 10Hz
        
        rospy.loginfo("4-Input Fuzzy Controller Node initialized")
    
    def road_info_callback(self, msg):
        """
        接收道路資訊回調
        
        預期格式: [road_detected, e_d, e_d_dot, e_l, e_l_dot, y_ground, x_ground]
        """
        if len(msg.data) < 5:
            rospy.logwarn_throttle(5.0, "road_info 格式不正確")
            return
        
        road_detected = bool(msg.data[0])
        
        # 檢查是否應該執行避障
        # 鍵盤控制優先，或者語音命令控制
        should_avoid = self.keyboard_enabled or (self.system_active and self.avoid_obstacle)
        
        if not road_detected or not should_avoid:
            self._publish_zero_velocity()
            return
        
        # 解析誤差值
        e_d = float(msg.data[1])
        e_d_dot = float(msg.data[2])
        e_l = float(msg.data[3])
        e_l_dot = float(msg.data[4])
        
        # 執行模糊推論（計時）
        infer_start = time.time()
        v, omega = self.controller.compute_with_logging(e_d, e_d_dot, e_l, e_l_dot)
        infer_time = (time.time() - infer_start) * 1000  # 毫秒
        
        # 記錄輸入輸出到 CSV
        with open(self.io_log_path, 'a', encoding='utf-8') as f:
            timestamp = time.time() - self.start_time  # 相對時間（秒）
            f.write(f"{timestamp:.3f},{e_d:.4f},{e_d_dot:.4f},{e_l:.4f},{e_l_dot:.4f},{v:.4f},{omega:.4f}\n")
        
        self.inference_times.append(infer_time)
        self.total_inference_time += infer_time
        self.inference_count += 1
        
        # 定期記錄統計
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            if self.inference_times:
                avg_time = sum(self.inference_times) / len(self.inference_times)
                total_avg = self.total_inference_time / self.inference_count if self.inference_count > 0 else 0
                
                rospy.loginfo(f"[Fuzzy Stats] Infer: avg={avg_time:.2f}ms | Total avg={total_avg:.2f}ms, count={self.inference_count}")
                
                # 寫入 log 檔案
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{time.strftime('%H:%M:%S')}, {avg_time:.2f}, {total_avg:.2f}, {self.inference_count}\n")
                
                self.inference_times.clear()
            self.last_log_time = current_time
        
        # 發布控制命令
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = omega
        self.cmd_vel_pub.publish(twist)
    
    def avoidance_control_callback(self, msg):
        """鍵盤避障控制回調"""
        if msg.data == "enabled":
            self.keyboard_enabled = True
            rospy.loginfo("Fuzzy Controller: 避障功能已啟動 (鍵盤控制)")
        else:
            self.keyboard_enabled = False
            self._publish_zero_velocity()
            rospy.loginfo("Fuzzy Controller: 避障功能已暫停 (鍵盤控制)")
    
    def system_status_callback(self, msg):
        """系統狀態回調"""
        status = msg.data
        if status == "voice_command_executing":
            self.system_active = False
        elif status == "fuzzy_control_active":
            if self.avoid_obstacle:
                self.system_active = True
    
    def voice_command_callback(self, msg):
        """語音命令回調"""
        command = msg.data.strip()
        
        if command == "21000":  # 開始任務
            self.task_started = True
            self.avoid_obstacle = False
        elif command == "10000":  # 停止
            self.avoid_obstacle = False
            self.system_active = False
            self.keyboard_enabled = False  # 語音停止也會停止鍵盤控制
        elif command.startswith(("11", "15", "16")):  # 移動指令
            if self.task_started:
                self.avoid_obstacle = True
    
    def _publish_zero_velocity(self):
        """發布零速度"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
    
    def run(self):
        """主循環"""
        rospy.loginfo("4-Input Fuzzy Controller running...")
        rospy.spin()


def test_fuzzy_controller():
    """測試模糊控制器"""
    print("\n===== 4-Input Fuzzy Controller Test =====\n")
    
    try:
        controller = FuzzyController4Input()
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        return
    
    # 測試案例
    test_cases = [
        # (e_d, e_d_dot, e_l, e_l_dot, 描述)
        (0.0, 0.0, 0.0, 0.0, "距離很近、置中、靜止"),
        (1.04, 0.0, 0.0, 0.0, "停止距離、置中、靜止"),
        (2.0, 0.0, 0.0, 0.0, "距離遠、置中、靜止"),
        (1.04, 0.0, -0.3, 0.0, "停止距離、左偏、靜止"),
        (1.04, 0.0, 0.3, 0.0, "停止距離、右偏、靜止"),
        (0.5, -1.0, 0.0, 0.0, "距離近、快速接近、置中"),
        (1.5, 1.0, 0.0, 0.0, "距離遠、快速遠離、置中"),
    ]
    
    print("模糊推論測試：")
    print("-" * 80)
    print(f"{'e_d':>6} {'e_d_dot':>8} {'e_l':>6} {'e_l_dot':>8} | {'v':>6} {'omega':>7} | 描述")
    print("-" * 80)
    
    for e_d, e_d_dot, e_l, e_l_dot, desc in test_cases:
        v, omega = controller.compute(e_d, e_d_dot, e_l, e_l_dot)
        print(f"{e_d:6.2f} {e_d_dot:8.2f} {e_l:6.2f} {e_l_dot:8.2f} | {v:6.3f} {omega:+7.3f} | {desc}")
    
    print("-" * 80)


def main():
    """主程式"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_fuzzy_controller()
    else:
        try:
            node = FuzzyControllerNode()
            node.run()
        except rospy.ROSInterruptException:
            pass


if __name__ == "__main__":
    main()
