#!/usr/bin/python3
"""
模糊控制資料記錄器
Fuzzy Control Data Logger

用途：記錄模糊控制器的所有輸入、輸出及系統狀態，便於後續分析調參。
輸出：CSV 檔案，包含時間戳、輸入誤差、輸出命令、濾波前後值等資訊。

使用方式：
  在另一個終端執行：rosrun yolov7_ros fuzzy_data_logger.py
  或加入 launch file 中一起啟動
"""

import os
import csv
import time
from datetime import datetime
from pathlib import Path

import rospy
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Twist


class FuzzyDataLogger:
    """模糊控制資料記錄器"""
    
    def __init__(self):
        """初始化記錄器"""
        rospy.init_node('fuzzy_data_logger', anonymous=True)
        
        # 建立輸出目錄和檔案
        self.output_dir = Path(rospy.get_param(
            '~output_dir', 
            '/home/cir/ros/src/yolov7_ros/src/fuzzy_control_design/logs'
        ))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 建立 CSV 檔案
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"fuzzy_data_{timestamp}.csv"
        
        # CSV 欄位定義
        self.fieldnames = [
            'timestamp',           # 時間戳（秒）
            'datetime',            # 可讀時間
            'road_detected',       # 是否檢測到道路
            'e_d',                 # 前方距離誤差
            'e_d_dot',             # 前方距離變化率
            'e_l',                 # 橫向誤差
            'e_l_dot',             # 橫向變化率
            'y_ground',            # 前方距離（公尺）
            'x_ground',            # 橫向距離（公尺）
            'fuzzy_v',             # 模糊控制線速度輸出
            'fuzzy_omega',         # 模糊控制角速度輸出
            'cmd_v',               # 實際發送線速度
            'cmd_omega',           # 實際發送角速度
            'avoidance_enabled',   # 避障是否啟用
            'system_status',       # 系統狀態
        ]
        
        # 初始化 CSV 檔案
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        
        rospy.loginfo(f"資料記錄檔案: {self.csv_path}")
        
        # 狀態變數
        self.road_info = {
            'road_detected': 0,
            'e_d': 0.0,
            'e_d_dot': 0.0,
            'e_l': 0.0,
            'e_l_dot': 0.0,
            'y_ground': 0.0,
            'x_ground': 0.0,
        }
        self.fuzzy_cmd = {'v': 0.0, 'omega': 0.0}
        self.actual_cmd = {'v': 0.0, 'omega': 0.0}
        self.avoidance_enabled = False
        self.system_status = "unknown"
        
        # 記錄開始時間
        self.start_time = time.time()
        self.record_count = 0
        
        # 訂閱 Topics
        rospy.Subscriber('/road_info', Float32MultiArray, self.road_info_callback)
        rospy.Subscriber('/fuzzy_cmd_vel', Twist, self.fuzzy_cmd_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        rospy.Subscriber('/avoidance_enabled', String, self.avoidance_callback)
        rospy.Subscriber('/system_status', String, self.status_callback)
        
        # 記錄頻率（10Hz，與控制迴路同步）
        self.rate = rospy.Rate(10)
        
        # 統計資訊
        self.stats = {
            'e_d_min': float('inf'), 'e_d_max': float('-inf'),
            'e_l_min': float('inf'), 'e_l_max': float('-inf'),
            'v_max': 0.0, 'omega_max': 0.0,
        }
        
        rospy.loginfo("模糊控制資料記錄器已啟動")
        rospy.loginfo("按 Ctrl+C 結束記錄並產生摘要報告")
    
    def road_info_callback(self, msg):
        """接收道路資訊"""
        if len(msg.data) >= 7:
            self.road_info = {
                'road_detected': int(msg.data[0]),
                'e_d': float(msg.data[1]),
                'e_d_dot': float(msg.data[2]),
                'e_l': float(msg.data[3]),
                'e_l_dot': float(msg.data[4]),
                'y_ground': float(msg.data[5]),
                'x_ground': float(msg.data[6]),
            }
            
            # 更新統計
            if self.road_info['road_detected']:
                self.stats['e_d_min'] = min(self.stats['e_d_min'], self.road_info['e_d'])
                self.stats['e_d_max'] = max(self.stats['e_d_max'], self.road_info['e_d'])
                self.stats['e_l_min'] = min(self.stats['e_l_min'], self.road_info['e_l'])
                self.stats['e_l_max'] = max(self.stats['e_l_max'], self.road_info['e_l'])
    
    def fuzzy_cmd_callback(self, msg):
        """接收模糊控制輸出"""
        self.fuzzy_cmd = {
            'v': msg.linear.x,
            'omega': msg.angular.z
        }
        self.stats['v_max'] = max(self.stats['v_max'], abs(msg.linear.x))
        self.stats['omega_max'] = max(self.stats['omega_max'], abs(msg.angular.z))
    
    def cmd_vel_callback(self, msg):
        """接收實際發送的控制命令"""
        self.actual_cmd = {
            'v': msg.linear.x,
            'omega': msg.angular.z
        }
    
    def avoidance_callback(self, msg):
        """避障狀態回調"""
        self.avoidance_enabled = (msg.data == "enabled")
    
    def status_callback(self, msg):
        """系統狀態回調"""
        self.system_status = msg.data
    
    def write_record(self):
        """寫入一筆記錄"""
        current_time = time.time()
        record = {
            'timestamp': f"{current_time - self.start_time:.3f}",
            'datetime': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'road_detected': self.road_info['road_detected'],
            'e_d': f"{self.road_info['e_d']:.4f}",
            'e_d_dot': f"{self.road_info['e_d_dot']:.4f}",
            'e_l': f"{self.road_info['e_l']:.4f}",
            'e_l_dot': f"{self.road_info['e_l_dot']:.4f}",
            'y_ground': f"{self.road_info['y_ground']:.4f}",
            'x_ground': f"{self.road_info['x_ground']:.4f}",
            'fuzzy_v': f"{self.fuzzy_cmd['v']:.4f}",
            'fuzzy_omega': f"{self.fuzzy_cmd['omega']:.4f}",
            'cmd_v': f"{self.actual_cmd['v']:.4f}",
            'cmd_omega': f"{self.actual_cmd['omega']:.4f}",
            'avoidance_enabled': int(self.avoidance_enabled),
            'system_status': self.system_status,
        }
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(record)
        
        self.record_count += 1
    
    def generate_summary(self):
        """產生摘要報告"""
        duration = time.time() - self.start_time
        
        summary_path = self.csv_path.with_suffix('.summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("模糊控制資料記錄摘要報告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"記錄時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"記錄長度: {duration:.1f} 秒\n")
            f.write(f"記錄筆數: {self.record_count} 筆\n")
            f.write(f"資料檔案: {self.csv_path}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("輸入變數範圍統計\n")
            f.write("-" * 60 + "\n")
            
            if self.stats['e_d_min'] != float('inf'):
                f.write(f"e_d (前方距離誤差): [{self.stats['e_d_min']:.4f}, {self.stats['e_d_max']:.4f}]\n")
                f.write(f"e_l (橫向誤差):     [{self.stats['e_l_min']:.4f}, {self.stats['e_l_max']:.4f}]\n")
            else:
                f.write("(未記錄到有效道路檢測資料)\n")
            
            f.write("\n")
            f.write("-" * 60 + "\n")
            f.write("輸出變數範圍統計\n")
            f.write("-" * 60 + "\n")
            f.write(f"v 最大值:     {self.stats['v_max']:.4f} m/s\n")
            f.write(f"omega 最大值: {self.stats['omega_max']:.4f} rad/s\n")
            
            f.write("\n")
            f.write("-" * 60 + "\n")
            f.write("調參建議\n")
            f.write("-" * 60 + "\n")
            
            # 根據記錄資料給出調參建議
            if self.stats['e_l_max'] - self.stats['e_l_min'] > 0.4:
                f.write("⚠ e_l 變化範圍較大，建議檢查橫向控制響應\n")
            if self.stats['omega_max'] < 0.3:
                f.write("⚠ omega 輸出較小，考慮增大模糊規則中 omega 的輸出等級\n")
            if self.stats['v_max'] < 0.2:
                f.write("⚠ v 輸出較小，考慮放寬線速度修正規則\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        rospy.loginfo(f"摘要報告已產生: {summary_path}")
        return summary_path
    
    def run(self):
        """主迴圈"""
        rospy.loginfo("開始記錄...")
        
        try:
            while not rospy.is_shutdown():
                self.write_record()
                
                # 每 100 筆顯示進度
                if self.record_count % 100 == 0:
                    rospy.loginfo(f"已記錄 {self.record_count} 筆資料")
                
                self.rate.sleep()
                
        except rospy.ROSInterruptException:
            pass
        finally:
            # 產生摘要報告
            self.generate_summary()
            rospy.loginfo(f"記錄完成，共 {self.record_count} 筆資料")
            rospy.loginfo(f"資料檔案: {self.csv_path}")


def main():
    """主函數"""
    try:
        logger = FuzzyDataLogger()
        logger.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
