#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 通訊延遲測試腳本

用途：精確測量 ROS Topic 的發佈/訂閱延遲

測試原理：
1. 發送帶時間戳的測試訊息到 cmd_vel
2. 訂閱 odom callback 時記錄接收時間
3. 計算 round-trip 延遲與單向估計延遲

執行方式: rosrun yolov7_ros ros_latency_test.py
或: python3 ros_latency_test.py

鍵盤控制：
  [W] 加速    [X] 減速
  [A] 左轉    [D] 右轉
  [S] 停止    [Q] 結束測試

輸出：延遲統計與時序圖
"""

import rospy
import time
import sys
import select
import termios
import tty
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import csv

# ---------------- 設定區 ----------------
SPEED_STEP = 0.05  # 每次按鍵的速度增量 (m/s)
TURN_STEP = 0.05   # 每次按鍵的轉向增量 (rad/s)
# ----------------------------------------


class ROSLatencyTester:
    def __init__(self):
        rospy.init_node('ros_latency_tester', anonymous=True)
        
        # Publishers & Subscribers
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # 數據存儲
        self.latencies = []  # 延遲記錄
        self.cmd_send_times = []  # 命令發送時間
        self.odom_recv_times = []  # odom 接收時間
        self.odom_header_stamps = []  # odom 訊息中的時間戳
        
        # 測試狀態
        self.last_cmd_time = None
        self.testing = False
        
        # 鍵盤控制：目標速度
        self.target_linear_x = 0.0
        self.target_angular_z = 0.0
        
        # 設定
        self.test_duration = 300.0  # 測試秒數（可提前按 Q 結束）
        self.test_rate = 10  # Hz
        
        # 保存終端機原始設定（為了鍵盤控制）
        self.settings = termios.tcgetattr(sys.stdin)
        
        print("ROS Latency Tester initialized")
        print(f"Max duration: {self.test_duration}s at {self.test_rate}Hz")
        print("Press [Q] to end test early")
    
    def get_key(self):
        """非阻塞式讀取鍵盤輸入"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def odom_callback(self, msg):
        """記錄 odom 接收時間"""
        recv_time = rospy.get_time()
        
        if self.testing and self.last_cmd_time is not None:
            # 計算從發送 cmd 到收到 odom 的延遲
            latency = (recv_time - self.last_cmd_time) * 1000  # ms
            
            if 0 < latency < 500:  # 過濾異常值
                self.latencies.append(latency)
                self.odom_recv_times.append(recv_time)
                
                # 檢查 odom header 時間戳
                if msg.header.stamp.to_sec() > 0:
                    self.odom_header_stamps.append(msg.header.stamp.to_sec())
    
    def run_test(self):
        """執行延遲測試（支援鍵盤控制）"""
        print("\n" + "="*50)
        print("Starting ROS Latency Test")
        print("="*50)
        print("Controls: [W]加速 [X]減速 [A]左轉 [D]右轉 [S]停止 [Q]結束")
        print("="*50)
        
        rate = rospy.Rate(self.test_rate)
        start_time = rospy.get_time()
        self.testing = True
        
        test_count = 0
        
        try:
            while not rospy.is_shutdown():
                elapsed = rospy.get_time() - start_time
                if elapsed > self.test_duration:
                    print("\nMax test duration reached.")
                    break
                
                # 讀取鍵盤輸入
                key = self.get_key()
                
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
                elif key == 'q' or key == '\x03':  # Q 或 Ctrl+C 結束
                    print("\nTest ended by user.")
                    break
                
                # 發送命令並記錄時間
                twist = Twist()
                twist.linear.x = self.target_linear_x
                twist.angular.z = self.target_angular_z
                
                self.last_cmd_time = rospy.get_time()
                self.cmd_send_times.append(self.last_cmd_time)
                self.pub_cmd.publish(twist)
                
                test_count += 1
                
                # 顯示狀態（每按一次鍵或每 20 次迴圈顯示）
                if key != '' or test_count % 20 == 0:
                    print(f"\r  Cmd: {self.target_linear_x:+.2f} m/s, {self.target_angular_z:+.2f} rad/s | "
                          f"Progress: {elapsed:.1f}s | Samples: {len(self.latencies)}    ", end='')
                
                rate.sleep()
        
        finally:
            # 確保機器人停止
            stop_twist = Twist()
            self.pub_cmd.publish(stop_twist)
            
            # 恢復終端機設定
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        
        self.testing = False
        print(f"\n\nTest completed. Total samples: {len(self.latencies)}")
        
        # 分析結果
        self.analyze_results()
    
    def analyze_results(self):
        """分析延遲結果"""
        # 診斷輸出：幫助了解資料收集狀況
        print("\n" + "-"*50)
        print("Diagnostic Info:")
        print(f"  Commands sent: {len(self.cmd_send_times)}")
        print(f"  Odom received: {len(self.odom_recv_times)}")
        print(f"  Valid latencies: {len(self.latencies)}")
        print("-"*50)
        
        if not self.latencies:
            print("\n[ERROR] No latency data collected!")
            print("Possible causes:")
            print("  1. /odom topic is not being published")
            print("  2. Test was interrupted before any odom received")
            print("  3. All latencies exceeded 500ms filter threshold")
            print("\nTip: Check if /odom is active with: rostopic hz /odom")
            return
        
        latencies = np.array(self.latencies)
        
        print("\n" + "="*50)
        print("ROS Communication Latency Results")
        print("="*50)
        print(f"Samples: {len(latencies)}")
        print(f"Mean latency: {np.mean(latencies):.2f} ms")
        print(f"Median latency: {np.median(latencies):.2f} ms")
        print(f"Min latency: {np.min(latencies):.2f} ms")
        print(f"Max latency: {np.max(latencies):.2f} ms")
        print(f"Std deviation: {np.std(latencies):.2f} ms")
        print(f"95th percentile: {np.percentile(latencies, 95):.2f} ms")
        print(f"99th percentile: {np.percentile(latencies, 99):.2f} ms")
        
        # 儲存結果
        self.save_results(latencies)
        self.plot_results(latencies)
    
    def save_results(self, latencies):
        """儲存結果到 CSV"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ros_latency_test_{timestamp_str}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'latency_ms'])
            for i, lat in enumerate(latencies):
                writer.writerow([i, round(lat, 2)])
        
        print(f"\nResults saved to: {filename}")
    
    def plot_results(self, latencies):
        """繪製延遲分佈圖"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 時序圖
        axes[0].plot(range(len(latencies)), latencies, 'b-', alpha=0.7, linewidth=0.8)
        axes[0].axhline(y=np.mean(latencies), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(latencies):.1f} ms')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('ROS cmd_vel → odom Latency Timeline')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 分佈直方圖
        axes[1].hist(latencies, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(x=np.mean(latencies), color='r', linestyle='--',
                       label=f'Mean: {np.mean(latencies):.1f} ms')
        axes[1].axvline(x=np.median(latencies), color='orange', linestyle='--',
                       label=f'Median: {np.median(latencies):.1f} ms')
        axes[1].set_xlabel('Latency (ms)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Latency Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f"ros_latency_plot_{timestamp_str}.png"
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        
        print(f"Plot saved to: {plot_filename}")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║          ROS Communication Latency Tester                 ║
╠══════════════════════════════════════════════════════════╣
║  This tool measures the latency between:                  ║
║    - Publishing to /cmd_vel                               ║
║    - Receiving response on /odom                          ║
║                                                           ║
║  Keyboard Controls:                                       ║
║    [W] Speed up   [X] Slow down                           ║
║    [A] Turn left  [D] Turn right                          ║
║    [S] Stop       [Q] End test                            ║
╚══════════════════════════════════════════════════════════╝
""")
    
    tester = None
    try:
        tester = ROSLatencyTester()
        rospy.sleep(1.0)  # Wait for connections
        tester.run_test()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        # 確保終端機設定恢復
        if tester is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, tester.settings)
            except Exception:
                pass


if __name__ == '__main__':
    main()
