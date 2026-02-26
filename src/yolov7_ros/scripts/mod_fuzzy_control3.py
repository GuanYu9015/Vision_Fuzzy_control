#!/usr/bin/python3
import rospy
import time
import math
import numpy as np
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def singleton(universe, value):
    """創建單一值模糊集"""
    return np.where(universe == value, 1.0, 0.0)

class FuzzyController:
    def __init__(self):
        # 初始化 ROS 節點
        rospy.init_node('fuzzy_controller_node', anonymous=True)
        
        # 創建訂閱者和發布者
        rospy.Subscriber('/road_info', Float32MultiArray, self.road_info_callback)
        rospy.Subscriber('/system_status', String, self.system_status_callback)
        rospy.Subscriber('/voice_command_code', String, self.voice_command_callback)
        
        # 發布到 /fuzzy_cmd_vel，而不是直接發布到 /cmd_vel
        self.cmd_vel_pub = rospy.Publisher('/fuzzy_cmd_vel', Twist, queue_size=10)
        
        # 初始化參數
        self.max_d_value = None  # 正規化的最大值
        self.max_d_dot = None
        self.previous_d_value = None
        self.previous_time = time.time()
        self.time_since_last_detection = 0.0
        
        # 系統狀態
        self.system_active = False  # 默認模糊控制是不活躍的，直到收到開始任務指令
        
        # 任務狀態追踪
        self.task_started = False   # 任務是否已開始
        self.avoid_obstacle = False # 是否啟用避障功能
        
        # 搜索相關參數
        self.searching = False
        self.search_direction = 'right'
        self.search_angle = 0.0
        self.max_search_angle = math.radians(30.0)
        self.angle_increment = math.radians(15.0)
        self.rotation_speed = 0.2
        self.total_search_time = 0.0
        self.max_total_search_time = 20.0
        
        # 是否已接收到第一個移動指令
        self.first_movement_received = False
        
        # 日誌顯示控制參數
        self.log_interval = rospy.get_param('~log_interval', 1.0)  # 預設每秒顯示一次
        self.last_log_time = time.time()
        
        # 發布一個默認的Twist消息
        self.publish_zero_velocity()
        
        # 設定模糊控制系統
        self.setup_fuzzy_control()
        
        # 設定 ROS 循環率
        self.rate = rospy.Rate(10)  # 10Hz
        
        rospy.loginfo("Fuzzy controller initialized and ready")
    
    def publish_zero_velocity(self):
        """發布零速度Twist消息"""
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_msg)
    
    def system_status_callback(self, msg):
        """系統狀態回調函數"""
        status = msg.data
        if status == "voice_command_executing":
            self.system_active = False
            rospy.loginfo("模糊控制已暫停，語音指令執行中")
        elif status == "fuzzy_control_active":
            # 修改：只在避障功能啟用時才激活模糊控制
            if self.avoid_obstacle:
                self.system_active = True
                rospy.loginfo("模糊控制已恢復活躍")
            else:
                # 避障功能未啟用時，保持非活躍狀態
                self.system_active = False
                rospy.loginfo("避障功能未啟用，模糊控制保持非活躍狀態")
        elif status == "error":
            rospy.logwarn("系統錯誤狀態")
    
    def voice_command_callback(self, msg):
        """接收語音命令的回調函數，用於檢測開始任務和移動指令"""
        command = msg.data.strip()
        
        # 檢查是否是開始任務指令
        if command == "21000":
            rospy.loginfo("模糊控制器接收到語音編碼：21000 (開始任務)")
            self.task_started = True
            self.avoid_obstacle = False  # 初始不啟用避障功能
            return
            
        # 檢查是否是停止指令
        if command == "10000":
            rospy.loginfo("模糊控制器接收到語音編碼：10000 (停止)")
            # 不改變 task_started 狀態，但關閉避障功能
            self.avoid_obstacle = False
            self.system_active = False
            return
            
        # 如果任務還未開始，忽略其他指令
        if not self.task_started:
            return
            
        # 檢查是否是移動相關的指令
        if command.startswith(("11", "15", "16")):
            rospy.loginfo(f"模糊控制器接收到語音編碼：{command} (移動指令)")
            # 啟用避障功能
            self.avoid_obstacle = True
            
            # 如果是第一個移動指令，記錄下來
            if not self.first_movement_received:
                rospy.loginfo("已接收到第一個移動指令，開始顯示速度資訊")
                self.first_movement_received = True
        
        # 檢查是否是純轉彎相關的指令 (13, 14開頭)
        elif command.startswith(("13", "14")):
            rospy.loginfo(f"模糊控制器接收到語音編碼：{command} (轉彎指令)")
            # 轉彎指令不影響避障功能的狀態
    
    def setup_fuzzy_control(self):
        """設定模糊控制系統"""
        try:
            # 定義模糊控制系統
            self.d = ctrl.Antecedent(np.arange(-0.3, 0.31, 0.01), 'd')
            self.d_dot = ctrl.Antecedent(np.arange(-0.2, 0.21, 0.01), 'd_dot')

            self.v = ctrl.Consequent(np.array([0.2, 0.3, 0.4]), 'v')
            self.w = ctrl.Consequent(np.array([-1.0, 0.0, 1.0]), 'w')

            # 定義輸入的模糊集
            self.d['N_h'] = fuzz.trapmf(self.d.universe, [-1, -0.4, -0.3, -0.2])
            self.d['N_m'] = fuzz.trimf(self.d.universe, [-0.3, -0.2, 0])
            self.d['Z'] = fuzz.trimf(self.d.universe, [-0.2, 0, 0.2])
            self.d['P_m'] = fuzz.trimf(self.d.universe, [0, 0.2, 0.3])
            self.d['P_h'] = fuzz.trapmf(self.d.universe, [0.2, 0.3, 0.4, 1])

            self.d_dot['N_h'] = fuzz.trapmf(self.d_dot.universe, [-1, -0.3, -0.2, -0.1])
            self.d_dot['N_m'] = fuzz.trimf(self.d_dot.universe, [-0.2, -0.1, 0])
            self.d_dot['Z'] = fuzz.trimf(self.d_dot.universe, [-0.1, 0, 0.1])
            self.d_dot['P_m'] = fuzz.trimf(self.d_dot.universe, [0, 0.1, 0.2])
            self.d_dot['P_h'] = fuzz.trapmf(self.d_dot.universe, [0.1, 0.2, 0.3, 1])

            # 定義輸出的模糊集（使用單一值模糊集）
            self.v['Slow'] = singleton(self.v.universe, 0.2)
            self.v['Medium'] = singleton(self.v.universe, 0.3)
            self.v['High'] = singleton(self.v.universe, 0.4)

            self.w['Negative'] = singleton(self.w.universe, -1.0)
            self.w['Zero'] = singleton(self.w.universe, 0.0)
            self.w['Positive'] = singleton(self.w.universe, 1.0)

            # 定義模糊規則
            rules = []
            
            # N_h rules
            rules.append(ctrl.Rule(self.d['N_h'] & self.d_dot['N_h'], (self.v['Slow'], self.w['Positive'])))
            rules.append(ctrl.Rule(self.d['N_h'] & self.d_dot['N_m'], (self.v['Medium'], self.w['Positive'])))
            rules.append(ctrl.Rule(self.d['N_h'] & self.d_dot['Z'], (self.v['High'], self.w['Positive'])))
            rules.append(ctrl.Rule(self.d['N_h'] & self.d_dot['P_m'], (self.v['Medium'], self.w['Positive'])))
            rules.append(ctrl.Rule(self.d['N_h'] & self.d_dot['P_h'], (self.v['Slow'], self.w['Positive'])))

            # N_m rules
            rules.append(ctrl.Rule(self.d['N_m'] & self.d_dot['N_h'], (self.v['Slow'], self.w['Positive'])))
            rules.append(ctrl.Rule(self.d['N_m'] & self.d_dot['N_m'], (self.v['Medium'], self.w['Positive'])))
            rules.append(ctrl.Rule(self.d['N_m'] & self.d_dot['Z'], (self.v['High'], self.w['Positive'])))
            rules.append(ctrl.Rule(self.d['N_m'] & self.d_dot['P_m'], (self.v['Medium'], self.w['Positive'])))
            rules.append(ctrl.Rule(self.d['N_m'] & self.d_dot['P_h'], (self.v['Slow'], self.w['Positive'])))

            # Z rules
            rules.append(ctrl.Rule(self.d['Z'] & self.d_dot['N_h'], (self.v['Medium'], self.w['Zero'])))
            rules.append(ctrl.Rule(self.d['Z'] & self.d_dot['N_m'], (self.v['Medium'], self.w['Zero'])))
            rules.append(ctrl.Rule(self.d['Z'] & self.d_dot['Z'], (self.v['High'], self.w['Zero'])))
            rules.append(ctrl.Rule(self.d['Z'] & self.d_dot['P_m'], (self.v['Medium'], self.w['Zero'])))
            rules.append(ctrl.Rule(self.d['Z'] & self.d_dot['P_h'], (self.v['Medium'], self.w['Zero'])))

            # P_m rules
            rules.append(ctrl.Rule(self.d['P_m'] & self.d_dot['N_h'], (self.v['Slow'], self.w['Negative'])))
            rules.append(ctrl.Rule(self.d['P_m'] & self.d_dot['N_m'], (self.v['Medium'], self.w['Negative'])))
            rules.append(ctrl.Rule(self.d['P_m'] & self.d_dot['Z'], (self.v['High'], self.w['Negative'])))
            rules.append(ctrl.Rule(self.d['P_m'] & self.d_dot['P_m'], (self.v['Medium'], self.w['Negative'])))
            rules.append(ctrl.Rule(self.d['P_m'] & self.d_dot['P_h'], (self.v['Slow'], self.w['Negative'])))

            # P_h rules
            rules.append(ctrl.Rule(self.d['P_h'] & self.d_dot['N_h'], (self.v['Slow'], self.w['Negative'])))
            rules.append(ctrl.Rule(self.d['P_h'] & self.d_dot['N_m'], (self.v['Medium'], self.w['Negative'])))
            rules.append(ctrl.Rule(self.d['P_h'] & self.d_dot['Z'], (self.v['High'], self.w['Negative'])))
            rules.append(ctrl.Rule(self.d['P_h'] & self.d_dot['P_m'], (self.v['Medium'], self.w['Negative'])))
            rules.append(ctrl.Rule(self.d['P_h'] & self.d_dot['P_h'], (self.v['Slow'], self.w['Negative'])))

            # 創建並初始化模糊控制系統
            self.fuzzy_ctrl = ctrl.ControlSystem(rules)
            self.fuzzy_sim = ctrl.ControlSystemSimulation(self.fuzzy_ctrl)
            
            rospy.loginfo("成功設置模糊控制系統")
        except Exception as e:
            rospy.logerr(f"設置模糊控制系統時出錯: {e}")
        
    def run(self):
        """主運行函數"""
        rospy.loginfo("Fuzzy controller running...")
        while not rospy.is_shutdown():
            self.rate.sleep()
            
    def road_info_callback(self, msg):
        """接收道路資訊的回調函數"""
        # 解析消息數據 [是否檢測到道路(0/1), 目標x座標, 圖像寬度]
        road_detected = bool(msg.data[0])
        current_time = time.time()
        delta_time = current_time - self.previous_time
        
        # 初始化輸出變量
        v_output = 0.0
        w_output = 0.0
        
        # 即使系統不活躍，也要顯示偵測到的道路資訊
        if road_detected:
            try:
                desired_x = float(msg.data[1])
                width = float(msg.data[2])
                
                # 如果 max_d_value 尚未設置，則設置為圖像寬度的一半
                if self.max_d_value is None:
                    self.max_d_value = width / 2
                    self.max_d_dot = width / 2
                    rospy.loginfo(f"max_d_value set to: {self.max_d_value}")
                    rospy.loginfo(f"max_d_dot set to: {self.max_d_dot}")
                    
                # 計算偏差值
                vehicle_center_x = width / 2  # 中心點
                d_value = desired_x - vehicle_center_x
                
                # 正規化水平距離
                normalized_d_value = d_value / self.max_d_value
                
                # 計算 d_dot (偏差變化率)
                if self.previous_d_value is not None and delta_time > 0:
                    d_dot_value = (d_value - self.previous_d_value) / delta_time
                else:
                    d_dot_value = 0
                    
                normalized_d_dot_value = d_dot_value / self.max_d_dot
                
                # 將 d 和 d_dot 值限制在各自的範圍內
                normalized_d_value = max(min(normalized_d_value, 0.3), -0.3)
                normalized_d_dot_value = max(min(normalized_d_dot_value, 0.2), -0.2)
                
                # 在任務已啟動後始終顯示正規化數值，無論是否啟用避障
                if self.task_started and (current_time - self.last_log_time >= self.log_interval):
                    rospy.loginfo(f"normalized_d_value: {normalized_d_value:.4f}")
                    rospy.loginfo(f"normalized_d_dot_value: {normalized_d_dot_value:.4f}")
                    self.last_log_time = current_time
                
                # 只有在系統活躍且避障功能啟用時才計算控制輸出
                if not self.system_active or not self.avoid_obstacle:
                    # 更新 previous_d_value
                    self.previous_d_value = d_value
                    # 更新 previous_time
                    self.previous_time = current_time
                    # 發布零速度命令
                    self.publish_zero_velocity()
                    return
                
                # 應用模糊控制
                self.fuzzy_sim.input['d'] = normalized_d_value
                self.fuzzy_sim.input['d_dot'] = normalized_d_dot_value
                
                try:
                    self.fuzzy_sim.compute()
                    
                    # 獲取模糊控制輸出
                    v_output = self.fuzzy_sim.output['v']
                    w_output = self.fuzzy_sim.output['w']
                    
                    # 限制輸出範圍
                    v_output = max(min(v_output, 0.4), 0.2)  # 保持 v 在 0.2 到 0.4 之間
                    w_output = max(min(w_output, 1.0), -1.0)  # 保持 w 在 -1 到 1 之間
                    
                except ValueError as e:
                    rospy.logerr(f"模糊計算錯誤: {e}")
                    # 設定默認值
                    v_output = 0.2
                    if normalized_d_value > 0:
                        w_output = -0.5
                    elif normalized_d_value < 0:
                        w_output = 0.5
                    else:
                        w_output = 0.0
                
                # 更新 previous_d_value
                self.previous_d_value = d_value
                
                # 重置搜索相關變量
                self.time_since_last_detection = 0.0
                self.searching = False
                self.search_angle = 0.0
                self.total_search_time = 0.0
                self.search_direction = 'right'
                
            except Exception as e:
                rospy.logerr(f"處理道路信息時出錯: {e}")
                v_output = 0.0
                w_output = 0.0
                
        else:
            # 未檢測到道路
            self.time_since_last_detection += delta_time
            
            if self.time_since_last_detection < 1.0:
                # 等待 1 秒鐘
                v_output = 0.0
                w_output = 0.0
            else:
                # 只有在系統活躍且避障功能啟用時才進入搜索模式
                if self.system_active and self.avoid_obstacle:
                    # 進入搜索模式
                    self.searching = True
                    self.total_search_time += delta_time
                    
                    if self.total_search_time >= self.max_total_search_time:
                        # 觸發緊急停止
                        v_output = 0.0
                        w_output = 0.0
                        rospy.logwarn("Emergency stop: Unable to find the road.")
                    else:
                        # 繼續搜索
                        v_output = 0.0  # 停止前進
                        
                        if self.search_angle < self.max_search_angle:
                            # 繼續當前方向的旋轉
                            if self.search_direction == 'right':
                                w_output = self.rotation_speed  # 向左旋轉
                            else:
                                w_output = -self.rotation_speed  # 向右旋轉
                                
                            # 更新已旋轉的角度
                            self.search_angle += abs(w_output) * delta_time
                        else:
                            # 已達到最大搜索角度，切換方向並重置搜索角度
                            self.search_angle = 0.0
                            if self.search_direction == 'right':
                                self.search_direction = 'left'
                            else:
                                self.search_direction = 'right'
                
                # 將 previous_d_value 設置為 None
                self.previous_d_value = None
                
        # 發布控制命令到 /fuzzy_cmd_vel
        twist_msg = Twist()
        twist_msg.linear.x = v_output
        twist_msg.angular.z = w_output
        self.cmd_vel_pub.publish(twist_msg)
        
        # 在任務已啟動後始終顯示速度資訊，無論是否啟用避障
        if self.task_started and (time.time() - self.last_log_time >= self.log_interval):
            rospy.loginfo(f"v: {v_output:.4f}, w: {w_output:.4f}")
            self.last_log_time = time.time()
        
        # 更新 previous_time
        self.previous_time = current_time
        
def main():
    try:
        controller = FuzzyController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Fuzzy controller node terminated.")
        pass

if __name__ == "__main__":
    main()
