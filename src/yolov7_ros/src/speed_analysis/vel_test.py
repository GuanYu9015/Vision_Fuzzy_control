#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import sys, select, termios, tty
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import math
from tf.transformations import euler_from_quaternion

# ---------------- 設定區 ----------------
CMD_TOPIC = '/cmd_vel'
ODOM_TOPIC = '/odom'       # 若你的機器人里程計話題不同，請修改這裡
IMU_TOPIC = '/imu'         # 若無 IMU 可忽略（訂閱不到就會維持預設值）

LOG_FILE = 'speed_test_result.csv'

MAX_TEST_SPEED = 5.0       # 線速度上限 (m/s) - 設高一點避免限制測試
SPEED_STEP = 0.05          # 每次按鍵增加 / 減少的線速度 (m/s)

MAX_TURN_SPEED = 5.0       # 角速度上限 (rad/s)
TURN_STEP = 0.05            # 每次按鍵增加 / 減少的角速度 (rad/s)
# ---------------------------------------

msg = """
---------------------------
最高速度測試記錄器 (ROS 1)
---------------------------
操作說明：
   w : 前進速度 + 0.05 m/s
   s : 後退速度 - 0.05 m/s
   a : 左轉角速度 + 0.05 rad/s
   d : 右轉角速度 - 0.05 rad/s

   x / 空白鍵 : 強制煞車 (線 & 角 速度歸零)

CTRL-C to quit
---------------------------
"""

class SpeedTestNode:
    def __init__(self):
        rospy.init_node('speed_test_logger', anonymous=True)
        
        # 1. 建立 Publisher 發送命令
        self.pub = rospy.Publisher(CMD_TOPIC, Twist, queue_size=1)
        
        # 2. 建立 Subscriber 接收數據
        self.sub_odom = rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_cb)
        self.sub_imu = rospy.Subscriber(IMU_TOPIC, Imu, self.imu_cb)
        
        # 指令速度狀態
        self.target_linear_x = 0.0
        self.target_angular_z = 0.0
        
        # ---- 感測器數值緩衝區 ----
        # Odometry
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.odom_lin_x = 0.0
        self.odom_lin_y = 0.0
        self.odom_lin_speed = 0.0
        self.odom_ang_z = 0.0

        # IMU
        self.imu_ang_x = 0.0
        self.imu_ang_y = 0.0
        self.imu_ang_z = 0.0
        self.imu_lin_x = 0.0
        self.imu_lin_y = 0.0
        self.imu_lin_z = 0.0
        
        # 3. 初始化 CSV 檔案（含完整欄位）
        self.csv_file = open(LOG_FILE, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'timestamp',
            'cmd_lin_x', 'cmd_ang_z',
            'odom_x', 'odom_y', 'odom_yaw',
            'odom_lin_x', 'odom_lin_y', 'odom_lin_speed', 'odom_ang_z',
            'imu_ang_x', 'imu_ang_y', 'imu_ang_z',
            'imu_lin_x', 'imu_lin_y', 'imu_lin_z'
        ])
        
        # 終端機設定 (用於讀取鍵盤)
        self.settings = termios.tcgetattr(sys.stdin)

    # ---------------- Odometry / IMU 回調 ----------------
    def odom_cb(self, msg: Odometry):
        """ 里程計回調函數：儲存位置、姿態、速度 """
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

        # 四元數轉 yaw
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat)
        self.odom_yaw = yaw

        # 速度
        self.odom_lin_x = msg.twist.twist.linear.x
        self.odom_lin_y = msg.twist.twist.linear.y
        self.odom_lin_speed = math.sqrt(self.odom_lin_x**2 + self.odom_lin_y**2)

        self.odom_ang_z = msg.twist.twist.angular.z

    def imu_cb(self, msg: Imu):
        """ IMU 回調函數 """
        self.imu_ang_x = msg.angular_velocity.x
        self.imu_ang_y = msg.angular_velocity.y
        self.imu_ang_z = msg.angular_velocity.z

        self.imu_lin_x = msg.linear_acceleration.x
        self.imu_lin_y = msg.linear_acceleration.y
        self.imu_lin_z = msg.linear_acceleration.z

    # ---------------- 鍵盤處理 ----------------
    def getKey(self):
        """ 讀取鍵盤輸入 (非阻塞模式) """
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1) # 等待 0.1 秒
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    # ---------------- 主迴圈 ----------------
    def run(self):
        print(msg)
        rate = rospy.Rate(10) # 設定迴圈頻率 10Hz (每秒記錄 10 次)
        
        try:
            while not rospy.is_shutdown():
                key = self.getKey()
                
                # --- 控制邏輯 (WASD 遞增 / 減) ---
                if key == 'w':   # 前進
                    if self.target_linear_x < MAX_TEST_SPEED:
                        self.target_linear_x += SPEED_STEP
                elif key == 's': # 後退
                    if self.target_linear_x > -MAX_TEST_SPEED:
                        self.target_linear_x -= SPEED_STEP
                elif key == 'a': # 左轉
                    if self.target_angular_z < MAX_TURN_SPEED:
                        self.target_angular_z += TURN_STEP
                elif key == 'd': # 右轉
                    if self.target_angular_z > -MAX_TURN_SPEED:
                        self.target_angular_z -= TURN_STEP
                elif key == 'x' or key == ' ':
                    self.target_linear_x = 0.0
                    self.target_angular_z = 0.0
                    print("!!! 緊急煞車 !!!")
                elif key == '\x03': # Ctrl+C (從 getKey 讀到)
                    break
                
                # --- 發送命令 ---
                twist = Twist()
                twist.linear.x = self.target_linear_x
                twist.angular.z = self.target_angular_z
                self.pub.publish(twist)

                # --- 顯示狀態 ---
                if key != '':
                    print(
                        f"指令: lin {self.target_linear_x:.2f} m/s, ang {self.target_angular_z:.2f} rad/s "
                        f"| Odom: lin {self.odom_lin_speed:.2f} m/s, ang {self.odom_ang_z:.2f} rad/s"
                    )

                # --- 記錄數據到 CSV ---
                self.writer.writerow([
                    rospy.get_time(),                        # 時間戳
                    round(self.target_linear_x, 3),          # 指令線速度
                    round(self.target_angular_z, 3),         # 指令角速度
                    round(self.odom_x, 4),                   # 里程計位置 x
                    round(self.odom_y, 4),                   # 里程計位置 y
                    round(self.odom_yaw, 4),                 # yaw (rad)
                    round(self.odom_lin_x, 4),               # odom 線速 x
                    round(self.odom_lin_y, 4),               # odom 線速 y
                    round(self.odom_lin_speed, 4),           # odom 合成線速
                    round(self.odom_ang_z, 4),               # odom 角速度 z
                    round(self.imu_ang_x, 4),                # IMU 角速度 x
                    round(self.imu_ang_y, 4),                # IMU 角速度 y
                    round(self.imu_ang_z, 4),                # IMU 角速度 z
                    round(self.imu_lin_x, 4),                # IMU 線加速度 x
                    round(self.imu_lin_y, 4),                # IMU 線加速度 y
                    round(self.imu_lin_z, 4),                # IMU 線加速度 z
                ])
                
                rate.sleep()

        except Exception as e:
            print(e)

        finally:
            # 結束前讓機器人停下來
            twist = Twist()
            self.pub.publish(twist)
            self.csv_file.close()
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            print(f"\n程式結束，數據已儲存至 {LOG_FILE}")

if __name__ == "__main__":
    node = SpeedTestNode()
    node.run()
