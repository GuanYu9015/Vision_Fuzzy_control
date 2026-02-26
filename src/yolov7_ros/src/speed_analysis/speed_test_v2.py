#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import sys, select, termios, tty
import math
from datetime import datetime  # <--- 新增這個模組
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
# 引入 tf 轉換庫 (若無安裝 tf 轉換庫可註解掉相關行)
from tf.transformations import euler_from_quaternion

# ---------------- 設定區 ----------------
CMD_TOPIC = '/cmd_vel'
ODOM_TOPIC = '/odom'
IMU_TOPIC = '/imu'   # 請確認這是您機器人的正確 Topic
JOINT_STATES_TOPIC = '/joint_states'  # 輪子編碼器/馬達狀態
MAX_TEST_SPEED = 5.0
SPEED_STEP = 0.05
TURN_STEP = 0.05
# ---------------------------------------

class SpeedTestNode:
    def __init__(self):
        rospy.init_node('speed_test_logger', anonymous=True)
        
        self.pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
        self.sub_odom = rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_cb)
        self.sub_imu = rospy.Subscriber(IMU_TOPIC, Imu, self.imu_cb)
        self.sub_joint = rospy.Subscriber(JOINT_STATES_TOPIC, JointState, self.joint_cb)
        
        # 控制目標
        self.target_linear_x = 0.0
        self.target_angular_z = 0.0
        
        # --- 數據緩衝區 ---
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.odom_lin_x = 0.0
        self.odom_lin_y = 0.0
        self.odom_ang_z = 0.0
        
        self.imu_ang_x = 0.0
        self.imu_ang_y = 0.0
        self.imu_ang_z = 0.0
        self.imu_lin_x = 0.0
        self.imu_lin_y = 0.0
        self.imu_lin_z = 0.0
        
        # --- Joint States 資料緩衝區 ---
        self.joint_names = []      # 關節名稱列表
        self.joint_positions = []  # 位置/角度 (rad)
        self.joint_velocities = [] # 速度 (rad/s)
        self.joint_efforts = []    # 力矩 (Nm)

        # --- 自動產生檔名 (日期_時間) ---
        # 格式範例: speed_test_20231025_143005.csv
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"speed_test_{current_time_str}.csv"
        
        print(f"準備建立日誌檔案: {self.log_filename}")
        
        self.csv_file = open(self.log_filename, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        
        # 寫入表頭
        self.writer.writerow([
            'timestamp', 
            'cmd_lin_x', 'cmd_ang_z', 
            'odom_x', 'odom_y', 'odom_yaw', 
            'odom_lin_x', 'odom_lin_y', 'odom_lin_speed', 'odom_ang_z',
            'imu_ang_x', 'imu_ang_y', 'imu_ang_z',
            'imu_lin_x', 'imu_lin_y', 'imu_lin_z',
            'joint_names', 'joint_positions', 'joint_velocities', 'joint_efforts'
        ])
        
        self.settings = termios.tcgetattr(sys.stdin)

    def odom_cb(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        
        # 四元數轉歐拉角
        try:
            orientation_q = msg.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
            self.odom_yaw = yaw
        except:
            self.odom_yaw = 0.0

        self.odom_lin_x = msg.twist.twist.linear.x
        self.odom_lin_y = msg.twist.twist.linear.y
        self.odom_ang_z = msg.twist.twist.angular.z

    def imu_cb(self, msg):
        self.imu_ang_x = msg.angular_velocity.x
        self.imu_ang_y = msg.angular_velocity.y
        self.imu_ang_z = msg.angular_velocity.z
        
        self.imu_lin_x = msg.linear_acceleration.x
        self.imu_lin_y = msg.linear_acceleration.y
        self.imu_lin_z = msg.linear_acceleration.z

    def joint_cb(self, msg):
        """處理 /joint_states 訊息，收集輪子編碼器與馬達狀態"""
        self.joint_names = list(msg.name)
        self.joint_positions = list(msg.position) if msg.position else []
        self.joint_velocities = list(msg.velocity) if msg.velocity else []
        self.joint_efforts = list(msg.effort) if msg.effort else []

    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        print("\n------------------------------------")
        print(f"記錄檔名: {self.log_filename}")
        print("操作: [W]加速  [X]減速  [A/D]轉向  [S]停止  [Ctrl+C]結束")
        print("------------------------------------")
        
        rate = rospy.Rate(10) # 10Hz
        
        try:
            while not rospy.is_shutdown():
                key = self.getKey()
                
                if key == 'w':
                    self.target_linear_x += SPEED_STEP
                elif key == 'x':
                    self.target_linear_x -= SPEED_STEP
                elif key == 'a':
                    self.target_angular_z += TURN_STEP
                elif key == 'd':
                    self.target_angular_z -= TURN_STEP
                elif key == 's' or key == ' ':
                    self.target_linear_x = 0.0
                    self.target_angular_z = 0.0
                elif key == '\x03':
                    break
                
                # 發布命令
                twist = Twist()
                twist.linear.x = self.target_linear_x
                twist.angular.z = self.target_angular_z
                self.pub.publish(twist)
                
                # 計算合速度
                current_speed = math.sqrt(self.odom_lin_x**2 + self.odom_lin_y**2)

                # 寫入 CSV
                # 將 joint_states 的列表轉為分號分隔字串，避免 CSV 欄位衝突
                joint_names_str = ';'.join(self.joint_names) if self.joint_names else ''
                joint_pos_str = ';'.join([f"{p:.4f}" for p in self.joint_positions]) if self.joint_positions else ''
                joint_vel_str = ';'.join([f"{v:.4f}" for v in self.joint_velocities]) if self.joint_velocities else ''
                joint_eff_str = ';'.join([f"{e:.4f}" for e in self.joint_efforts]) if self.joint_efforts else ''
                
                self.writer.writerow([
                    f"{rospy.get_time():.4f}",
                    round(self.target_linear_x, 3),
                    round(self.target_angular_z, 3),
                    
                    round(self.odom_x, 4),
                    round(self.odom_y, 4),
                    round(self.odom_yaw, 4),
                    
                    round(self.odom_lin_x, 4),
                    round(self.odom_lin_y, 4),
                    round(current_speed, 4),
                    round(self.odom_ang_z, 4),
                    
                    round(self.imu_ang_x, 4),
                    round(self.imu_ang_y, 4),
                    round(self.imu_ang_z, 4),
                    
                    round(self.imu_lin_x, 4),
                    round(self.imu_lin_y, 4),
                    round(self.imu_lin_z, 4),
                    
                    joint_names_str,
                    joint_pos_str,
                    joint_vel_str,
                    joint_eff_str
                ])
                
                if key != '':
                    print(f"Cmd: {self.target_linear_x:.2f} m/s | Odom: {self.odom_lin_x:.2f} m/s")

                rate.sleep()

        except Exception as e:
            print(f"Error: {e}")

        finally:
            twist = Twist()
            self.pub.publish(twist)
            self.csv_file.close()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            print(f"\n程式結束，數據已儲存至: {self.log_filename}")

if __name__ == "__main__":
    node = SpeedTestNode()
    node.run()