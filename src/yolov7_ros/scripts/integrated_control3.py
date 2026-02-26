#!/usr/bin/python3

import rospy
import time
from std_msgs.msg import Float32MultiArray, String, Bool
from geometry_msgs.msg import Twist

class IntegratedController:
    def __init__(self):
        # 初始化ROS節點
        rospy.init_node('integrated_controller_node', anonymous=True)
        
        # 系統狀態與語音控制相關參數
        self.is_listening_voice = True       # 是否聽取語音輸入
        self.is_executing_voice_cmd = False  # 是否正在執行語音指令
        self.paused = False                  # 是否處於完全暫停狀態
        self.last_voice_cmd_time = 0         # 上一次接收語音指令的時間
        self.voice_cmd_timeout = 25.0        # 語音指令最大執行時間（秒），防止無限執行
        
        # 動作執行狀態跟踪
        self.cmd_executing = False           # 是否有具體動作正在執行
        self.cmd_start_time = 0              # 動作開始時間
        self.cmd_duration = 0                # 預計動作持續時間
        self.current_cmd = None              # 當前正在執行的指令編碼
        
        # 新增：任務啟動標誌
        self.task_started = False            # 任務是否已經啟動
        self.observation_completed = False   # 是否已完成左右觀察動作
        self.avoid_obstacle = False          # 是否啟用避障功能
        # 修改：移除 first_forward_received 標誌
        # self.first_forward_received = True  # 是否已收到第一個前進指令
        
        # 模糊控制輸出暫存
        self.fuzzy_cmd = Twist()
        self.road_detected = False
        
        # 創建發布者
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.system_status_pub = rospy.Publisher('/system_status', String, queue_size=10)
        
        # 創建訂閱者
        rospy.Subscriber('/road_info', Float32MultiArray, self.road_info_callback)
        rospy.Subscriber('/fuzzy_cmd_vel', Twist, self.fuzzy_cmd_callback)
        rospy.Subscriber('/voice_command_code', String, self.voice_command_callback)
        
        # 設定執行頻率
        self.rate = rospy.Rate(10)  # 10Hz
        
        rospy.loginfo("整合控制器已初始化完成")
        
    def road_info_callback(self, msg):
        """接收道路信息的回調"""
        self.road_detected = bool(msg.data[0])
    
    def fuzzy_cmd_callback(self, msg):
        """接收模糊控制輸出的回調"""
        # 儲存模糊控制系統的輸出
        self.fuzzy_cmd = msg
    
    def voice_command_callback(self, msg):
        """接收語音命令的回調"""
        if not self.is_listening_voice:
            rospy.loginfo("系統目前不接受語音指令，指令已忽略")
            return
            
        command_text = msg.data.strip()
        
        if len(command_text) < 5:
            rospy.logwarn("接收到的編碼長度不足")
            return
            
        # 取五位數字作為指令
        cmd = command_text[0:5]
        rospy.loginfo(f"整合控制器接收到語音編碼：{cmd}")
        
        # 檢查是否接收到「開始任務」指令 (21000)
        if cmd == "21000":
            # 重設執行狀態，確保命令能被執行
            if self.cmd_executing:
                rospy.loginfo("中斷當前正在執行的指令，執行新指令")
                self.stop_robot()
                self.cmd_executing = False
                
            self.is_executing_voice_cmd = True
            self.last_voice_cmd_time = rospy.Time.now().to_sec()
            self.system_status_pub.publish("voice_command_executing")
            
            # 設置當前命令
            self.current_cmd = cmd
            
            if not self.task_started:
                rospy.loginfo("收到「開始任務」指令，不啟用避障功能")
                self.task_started = True
                self.avoid_obstacle = False
            else:
                rospy.loginfo("已經在執行任務中，重新執行左右觀察")
                
            # 如果系統處於暫停狀態，復位暫停狀態
            if self.paused:
                rospy.loginfo("從暫停狀態中恢復")
                self.paused = False
                
            # 執行左右觀察動作
            self.execute_command(self.left_right_observation)
            return
        
        # 檢查是否接收到「停止」指令 (10000)
        if cmd == "10000":
            rospy.loginfo("接收到「停止」指令，系統將完全暫停等待下一個指令")
            self.paused = True
            self.is_executing_voice_cmd = True
            self.last_voice_cmd_time = rospy.Time.now().to_sec()
            self.system_status_pub.publish("voice_command_executing")
            self.stop_robot()
            
            # 重置動作執行狀態
            self.cmd_executing = False
            self.current_cmd = None
            return
        
        # 檢查任務是否已啟動
        if not self.task_started:
            rospy.loginfo("尚未收到「開始任務」指令，忽略其他指令")
            return
        
        # 檢查是否為移動相關指令(11, 15, 16開頭)
        if cmd.startswith(("11", "15", "16")):
            # 設置避障功能為啟用
            if not self.avoid_obstacle:
                self.avoid_obstacle = True
                rospy.loginfo("收到移動指令，啟用避障功能")
                # 通知系統狀態，確保模糊控制知道避障已啟用
                self.system_status_pub.publish("fuzzy_control_active")
        
        # 如果系統處於暫停狀態，收到其他指令後恢復
        if self.paused:
            rospy.loginfo("從暫停狀態中恢復")
            self.paused = False
        
        # 如果當前有指令在執行中，先停止當前動作
        if self.cmd_executing:
            rospy.loginfo("中斷當前正在執行的指令，執行新指令")
            self.stop_robot()
            self.cmd_executing = False
        
        self.is_executing_voice_cmd = True
        self.last_voice_cmd_time = rospy.Time.now().to_sec()
        self.system_status_pub.publish("voice_command_executing")
        self.current_cmd = cmd
        
        # 根據指令執行相應動作
        if cmd.startswith("13"):  # 左轉指令
            if cmd == "13000":  # 左轉90度
                self.execute_command(self.turn_left)
            elif len(cmd) == 5 and cmd[0:3] == "131":  # 左轉特定角度
                try:
                    degrees = int(cmd[3:5])
                    self.execute_command(lambda: self.turn_left_degrees(degrees))
                except ValueError:
                    rospy.logwarn(f"無效的左轉角度值：{cmd[3:5]}")
                    self.is_executing_voice_cmd = False
        elif cmd.startswith("14"):  # 右轉指令
            if cmd == "14000":  # 右轉90度
                self.execute_command(self.turn_right)
            elif len(cmd) == 5 and cmd[0:3] == "141":  # 右轉特定角度
                try:
                    degrees = int(cmd[3:5])
                    self.execute_command(lambda: self.turn_right_degrees(degrees))
                except ValueError:
                    rospy.logwarn(f"無效的右轉角度值：{cmd[3:5]}")
                    self.is_executing_voice_cmd = False
        elif cmd.startswith("11"):  # 前進指令
            if cmd == "11000":  # 一直前進
                self.execute_command(self.move_forward)
            elif len(cmd) == 5 and cmd[0:3] == "111":  # 前進特定距離
                try:
                    distance = int(cmd[3:5])
                    self.execute_command(lambda: self.move_forward_distance(distance))
                except ValueError:
                    rospy.logwarn(f"無效的前進距離值：{cmd[3:5]}")
                    self.is_executing_voice_cmd = False
        elif cmd.startswith("15"):  # 左轉後前進
            if cmd == "15000":  # 左轉90度後前進
                self.execute_command(self.turn_left_then_forward)
            elif len(cmd) == 5 and cmd[0:3] == "152" and cmd[3] == "0":  # 左轉到特定位置
                try:
                    location = int(cmd[4])
                    self.execute_command(lambda: self.turn_left_to_location(location))
                except ValueError:
                    rospy.logwarn(f"無效的位置值：{cmd[4]}")
                    self.is_executing_voice_cmd = False
        elif cmd.startswith("16"):  # 右轉後前進
            if cmd == "16000":  # 右轉90度後前進
                self.execute_command(self.turn_right_then_forward)
            elif len(cmd) == 5 and cmd[0:3] == "162" and cmd[3] == "0":  # 右轉到特定位置
                try:
                    location = int(cmd[4])
                    self.execute_command(lambda: self.turn_right_to_location(location))
                except ValueError:
                    rospy.logwarn(f"無效的位置值：{cmd[4]}")
                    self.is_executing_voice_cmd = False
        elif cmd.startswith("112") and cmd[3] == "0":  # 前進到特定位置
            try:
                location = int(cmd[4])
                self.execute_command(lambda: self.move_forward_to_location(location))
            except ValueError:
                rospy.logwarn(f"無效的位置值：{cmd[4]}")
                self.is_executing_voice_cmd = False
        else:
            rospy.logwarn(f"未知的指令編碼：{cmd}")
            self.is_executing_voice_cmd = False
    
    # def execute_command(self, command_func):
    #     """啟動指令的執行 - 只是設置標誌位，實際執行在主循環中"""
    #     try:
    #         # 設置開始執行指令的標誌
    #         self.cmd_executing = True
    #         self.cmd_start_time = rospy.Time.now().to_sec()
            
    #         # 執行指令函數（只會設置參數，不會阻塞）
    #         command_func()
            
    #         rospy.loginfo(f"指令 {self.current_cmd} 開始執行，預計持續 {self.cmd_duration} 秒")
            
    #     except Exception as e:
    #         rospy.logerr(f"指令執行錯誤：{e}")
    #         self.cmd_executing = False
    #         self.is_executing_voice_cmd = False
    #         self.system_status_pub.publish("error")
    
        # ── 原始寫法 (會先開啟執行旗標，cmd_duration 還是 0) ──
    # self.cmd_executing = True
    # self.cmd_start_time = rospy.Time.now().to_sec()
    # command_func()                        # left_right_observation() 才設定 7.0
    # …

    def execute_command(self, command_func):
        """啟動指令的執行（修正版，先執行命令函數再設置標志位）"""
        try:
            # 先執行命令函數來設置所需的參數（如cmd_duration）
            command_func()
            
            # 然後設置執行標志位和開始時間
            self.cmd_executing = True
            self.cmd_start_time = rospy.Time.now().to_sec()
            
            rospy.loginfo(f"指令 {self.current_cmd} 開始執行，預計持續 {self.cmd_duration} 秒")
            
        except Exception as e:
            rospy.logerr(f"指令執行錯誤：{e}")
            self.cmd_executing = False
            self.is_executing_voice_cmd = False
            self.system_status_pub.publish("error")

    
    def check_command_completion(self):
        """檢查當前執行的命令是否已完成"""
        if not self.cmd_executing:
            return
            
        current_time = rospy.Time.now().to_sec()
        elapsed_time = current_time - self.cmd_start_time
        
        # 如果動作已執行完預定時間，標記為完成
        if elapsed_time >= self.cmd_duration:
            rospy.loginfo(f"指令 {self.current_cmd} 已完成執行")

            self.stop_robot()
            self.cmd_executing = False
            
            # 如果是左右觀察指令完成，設置觀察已完成
            if self.current_cmd == "21000":
                self.observation_completed = True
                rospy.loginfo("左右觀察已完成，等待移動指令")
            
            self.current_cmd = None
            
            # 執行完畢後恢復模糊控制，確保避障能力生效
            self.is_executing_voice_cmd = False
            self.system_status_pub.publish("fuzzy_control_active")
            rospy.loginfo("語音指令執行完畢，恢復模糊控制")
            
            # 確保模糊控制知道避障狀態
            if self.avoid_obstacle:
                rospy.loginfo("確保避障功能保持啟用狀態")
    
    def stop_robot(self):
        """停止機器人"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo("機器人已停止")
    
    def left_right_observation(self):
        """左右觀察環境（先左90度，再右180度，再左90度，回到中間）"""
        rospy.loginfo("執行左右觀察：左旋轉90度 -> 右旋轉180度 -> 左旋轉90度回到中間")
        
        # 設定總預計執行時間
        self.cmd_duration = 8.35  # 1.57 + 1.0 + 3.14 + 1.0 + 1.57 秒
        
        # 此函數不再直接執行動作，而是返回讓主循環負責執行
        # 執行邏輯移至 process_left_right_observation 方法
    
    def process_left_right_observation(self):
        """在主循環中處理左右觀察的執行邏輯"""
        elapsed_time = rospy.Time.now().to_sec() - self.cmd_start_time
        
        if elapsed_time < 1.6:  # 左旋轉階段
            twist = Twist()
            twist.angular.z = 1.0  # 左轉
            self.cmd_vel_pub.publish(twist)
        elif elapsed_time < 2.6:  # 第一次暫停
            self.stop_robot()
        elif elapsed_time < 5.78:  # 右旋轉階段 (2.57 + 3.14)
            twist = Twist()
            twist.angular.z = -1.0  # 右轉
            self.cmd_vel_pub.publish(twist)
        elif elapsed_time < 6.82:  # 第二次暫停
            self.stop_robot()
        elif elapsed_time < 8.35:        # 左旋回中心
            twist = Twist()
            twist.angular.z = 1.0
            self.cmd_vel_pub.publish(twist)
        else:                           # 最後 0.2 s 強制停止
            self.stop_robot()
    
    def turn_left(self):
        """執行左轉90度"""
        rospy.loginfo("執行左轉90度")
        self.cmd_duration = 1.57  # 約90度的旋轉時間
    
    def process_turn_left(self):
        """在主循環中處理左轉的執行邏輯"""
        twist = Twist()
        twist.angular.z = 1.0  # 左轉
        self.cmd_vel_pub.publish(twist)
    
    def turn_left_degrees(self, degrees):
        """執行左轉特定角度"""
        rospy.loginfo(f"執行左轉{degrees}度")
        self.cmd_duration = degrees * 0.0174533  # 角度轉弧度
    
    def process_turn_left_degrees(self):
        """在主循環中處理左轉特定角度的執行邏輯"""
        twist = Twist()
        twist.angular.z = 1.0  # 左轉
        self.cmd_vel_pub.publish(twist)
    
    def turn_right(self):
        """執行右轉90度"""
        rospy.loginfo("執行右轉90度")
        self.cmd_duration = 1.57  # 約90度的旋轉時間
    
    def process_turn_right(self):
        """在主循環中處理右轉的執行邏輯"""
        twist = Twist()
        twist.angular.z = -1.0  # 右轉
        self.cmd_vel_pub.publish(twist)
    
    def turn_right_degrees(self, degrees):
        """執行右轉特定角度"""
        rospy.loginfo(f"執行右轉{degrees}度")
        self.cmd_duration = degrees * 0.0174533  # 角度轉弧度
    
    def process_turn_right_degrees(self):
        """在主循環中處理右轉特定角度的執行邏輯"""
        twist = Twist()
        twist.angular.z = -1.0  # 右轉
        self.cmd_vel_pub.publish(twist)
    
    def move_forward(self):
        """執行前進，同時啟用障礙物迴避"""
        rospy.loginfo("執行前進（帶避障）")
        self.cmd_duration = 1.0  # 前進持續時間
    
    def process_move_forward(self):
        """在主循環中處理前進的執行邏輯，結合避障"""
        twist = Twist()
        twist.linear.x = 0.3  # 固定前進速度
        
        # 結合模糊控制輸出進行避障
        if self.road_detected and self.avoid_obstacle:
            twist.angular.z = self.fuzzy_cmd.angular.z  # 使用模糊控制的轉向
            rospy.loginfo(f"避障功能已啟用，使用模糊控制，angular.z = {self.fuzzy_cmd.angular.z}")
        
        self.cmd_vel_pub.publish(twist)
    
    def move_forward_distance(self, distance):
        """執行前進特定距離（單位：米）"""
        rospy.loginfo(f"執行前進{distance}米")
        # 假設速度為0.3m/s，計算需要的時間
        self.cmd_duration = distance / 0.3
    
    def process_move_forward_distance(self):
        """在主循環中處理前進特定距離的執行邏輯，結合避障"""
        twist = Twist()
        twist.linear.x = 0.3  # 固定前進速度
        
        # 結合模糊控制輸出進行避障
        if self.road_detected and self.avoid_obstacle:
            twist.angular.z = self.fuzzy_cmd.angular.z  # 使用模糊控制的轉向
            rospy.loginfo(f"避障功能已啟用，使用模糊控制，angular.z = {self.fuzzy_cmd.angular.z}")
        
        self.cmd_vel_pub.publish(twist)
    
    def move_forward_to_location(self, location):
        """前進到特定位置"""
        rospy.loginfo(f"執行前進到位置：{self.get_location_name(location)}")
        self.cmd_duration = 10.0  # 假設前進10秒到達目標位置
    
    def process_move_forward_to_location(self):
        """在主循環中處理前進到特定位置的執行邏輯，結合避障"""
        twist = Twist()
        twist.linear.x = 0.3  # 固定前進速度
        
        # 結合模糊控制輸出進行避障
        if self.road_detected and self.avoid_obstacle:
            twist.angular.z = self.fuzzy_cmd.angular.z  # 使用模糊控制的轉向
            rospy.loginfo(f"避障功能已啟用，使用模糊控制，angular.z = {self.fuzzy_cmd.angular.z}")
        
        self.cmd_vel_pub.publish(twist)
    
    def turn_left_to_location(self, location):
        """左轉90度後前進到特定位置"""
        rospy.loginfo(f"執行左轉90度後前進到位置：{self.get_location_name(location)}")
        self.cmd_duration = 11.57  # 左轉90度(1.57秒) + 前進10秒
    
    def process_turn_left_to_location(self):
        """在主循環中處理左轉後前進到位置的執行邏輯"""
        elapsed_time = rospy.Time.now().to_sec() - self.cmd_start_time
        
        if elapsed_time < 1.57:  # 左轉階段
            twist = Twist()
            twist.angular.z = 1.0  # 左轉
            self.cmd_vel_pub.publish(twist)
        else:  # 前進階段
            twist = Twist()
            twist.linear.x = 0.3  # 固定前進速度
            
            # 結合模糊控制輸出進行避障
            if self.road_detected and self.avoid_obstacle:
                twist.angular.z = self.fuzzy_cmd.angular.z  # 使用模糊控制的轉向
                rospy.loginfo(f"避障功能已啟用，使用模糊控制，angular.z = {self.fuzzy_cmd.angular.z}")
            
            self.cmd_vel_pub.publish(twist)
    
    def turn_right_to_location(self, location):
        """右轉90度後前進到特定位置"""
        rospy.loginfo(f"執行右轉90度後前進到位置：{self.get_location_name(location)}")
        self.cmd_duration = 11.57  # 右轉90度(1.57秒) + 前進10秒
    
    def process_turn_right_to_location(self):
        """在主循環中處理右轉後前進到位置的執行邏輯"""
        elapsed_time = rospy.Time.now().to_sec() - self.cmd_start_time
        
        if elapsed_time < 1.57:  # 右轉階段
            twist = Twist()
            twist.angular.z = -1.0  # 右轉
            self.cmd_vel_pub.publish(twist)
        else:  # 前進階段
            twist = Twist()
            twist.linear.x = 0.3  # 固定前進速度
            
            # 結合模糊控制輸出進行避障
            if self.road_detected and self.avoid_obstacle:
                twist.angular.z = self.fuzzy_cmd.angular.z  # 使用模糊控制的轉向
                rospy.loginfo(f"避障功能已啟用，使用模糊控制，angular.z = {self.fuzzy_cmd.angular.z}")
            
            self.cmd_vel_pub.publish(twist)
    
    def turn_left_then_forward(self):
        """左轉90度後前進"""
        rospy.loginfo("執行左轉90度後前進")
        self.cmd_duration = 1.6  # 左轉90度(1.57秒) + 前進3秒
    
    def process_turn_left_then_forward(self):
        """在主循環中處理左轉後前進的執行邏輯"""
        elapsed_time = rospy.Time.now().to_sec() - self.cmd_start_time
        
        if elapsed_time < 1.57:  # 左轉階段
            twist = Twist()
            twist.angular.z = 1.0  # 左轉
            self.cmd_vel_pub.publish(twist)
        else:  # 前進階段
            twist = Twist()
            twist.linear.x = 0.3  # 固定前進速度
            
            # 結合模糊控制輸出進行避障
            if self.road_detected and self.avoid_obstacle:
                twist.angular.z = self.fuzzy_cmd.angular.z  # 使用模糊控制的轉向
                rospy.loginfo(f"避障功能已啟用，使用模糊控制，angular.z = {self.fuzzy_cmd.angular.z}")
            
            self.cmd_vel_pub.publish(twist)
    
    def turn_right_then_forward(self):
        """右轉90度後前進"""
        rospy.loginfo("執行右轉90度後前進")
        self.cmd_duration = 1.6  # 右轉90度(1.57秒) + 前進3秒
    
    def process_turn_right_then_forward(self):
        """在主循環中處理右轉後前進的執行邏輯"""
        elapsed_time = rospy.Time.now().to_sec() - self.cmd_start_time
        
        if elapsed_time < 1.57:  # 右轉階段
            twist = Twist()
            twist.angular.z = -1.0  # 右轉
            self.cmd_vel_pub.publish(twist)
        else:  # 前進階段
            twist = Twist()
            twist.linear.x = 0.3  # 固定前進速度
            
            # 結合模糊控制輸出進行避障
            if self.road_detected and self.avoid_obstacle:
                twist.angular.z = self.fuzzy_cmd.angular.z  # 使用模糊控制的轉向
                rospy.loginfo(f"避障功能已啟用，使用模糊控制，angular.z = {self.fuzzy_cmd.angular.z}")
            
            self.cmd_vel_pub.publish(twist)
    
    def get_location_name(self, location_code):
        """根據位置代碼獲取位置名稱"""
        locations = {
            1: "傷患",
            2: "斜坡",
            3: "門口",
            4: "轉彎處",
            5: "廁所",
            6: "電梯",
            7: "樓梯"
        }
        return locations.get(location_code, f"未知位置({location_code})")
    
    def process_current_command(self):
        """根據當前指令類型調用對應的處理方法"""
        if not self.cmd_executing or self.current_cmd is None:
            return
            
        if self.current_cmd == "21000":
            self.process_left_right_observation()
        elif self.current_cmd == "13000":
            self.process_turn_left()
        elif self.current_cmd.startswith("131"):
            self.process_turn_left_degrees()
        elif self.current_cmd == "14000":
            self.process_turn_right()
        elif self.current_cmd.startswith("141"):
            self.process_turn_right_degrees()
        elif self.current_cmd == "11000":
            self.process_move_forward()
        elif self.current_cmd.startswith("111"):
            self.process_move_forward_distance()
        elif self.current_cmd.startswith("1120"):
            self.process_move_forward_to_location()
        elif self.current_cmd == "15000":
            self.process_turn_left_then_forward()
        elif self.current_cmd.startswith("1520"):
            self.process_turn_left_to_location()
        elif self.current_cmd == "16000":
            self.process_turn_right_then_forward()
        elif self.current_cmd.startswith("1620"):
            self.process_turn_right_to_location()
    
    def check_voice_cmd_timeout(self):
        """檢查語音指令是否超時（暫停狀態下不自動恢復）"""
        if self.is_executing_voice_cmd and not self.paused:
            current_time = rospy.Time.now().to_sec()
            if current_time - self.last_voice_cmd_time > self.voice_cmd_timeout:
                rospy.logwarn("語音指令執行超時，強制恢復模糊控制")
                self.is_executing_voice_cmd = False
                self.cmd_executing = False
                self.system_status_pub.publish("fuzzy_control_active")
                self.stop_robot()
    
    def run(self):
        """主循環"""
        rospy.loginfo("整合控制系統啟動中...")
        
        while not rospy.is_shutdown():
            # 檢查語音指令是否超時（安全機制）
            self.check_voice_cmd_timeout()
            
            # 檢查當前命令是否已完成
            self.check_command_completion()
            
            # 三種運行狀態：
            # 1. 完全暫停 - 不執行任何動作
            # 2. 執行語音指令 - 根據指令類型執行對應動作，同時保持避障能力
            # 3. 正常模糊控制 - 完全由模糊控制系統引導
            
            if self.paused:
                # 暫停狀態，不執行任何動作
                pass
            elif self.is_executing_voice_cmd and self.cmd_executing:
                # 如果正在執行具體的語音指令動作
                self.process_current_command()
            elif not self.is_executing_voice_cmd and self.task_started and self.avoid_obstacle:
                # 當未在執行語音指令模式且任務已啟動且避障已啟用時，採用模糊控制命令
                # 確保模糊控制命令能夠控制機器人
                if self.road_detected:
                    rospy.loginfo_throttle(1.0, f"處於模糊控制模式，命令: linear.x={self.fuzzy_cmd.linear.x}, angular.z={self.fuzzy_cmd.angular.z}")
                    self.cmd_vel_pub.publish(self.fuzzy_cmd)
                else:
                    rospy.loginfo_throttle(1.0, "未檢測到道路，停止移動")
                    self.stop_robot()
            
            self.rate.sleep()

if __name__ == "__main__":
    try:
        controller = IntegratedController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("整合控制節點已終止")
