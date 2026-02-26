#!/usr/bin/env python3
"""
ROS 資源監控節點 (Resource Monitor Node)

基於 tegrastats 監控 NVIDIA Jetson 的系統資源使用狀況，
並將數據記錄到 CSV 檔案中。在 roslaunch 啟動時自動開始監控。

使用 tegrastats 而非 jtop，可更準確捕捉 GPU 記憶體與功耗資料。
"""
import csv
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import rospy


class TegrastatsParser:
    """
    解析 tegrastats 輸出的工具類別
    
    tegrastats 輸出範例：
    RAM 5164/15523MB SWAP 0/7761MB CPU [8%@1113,6%@1113,...] GR3D_FREQ 0%@[0] ...
    """
    
    @staticmethod
    def parse_line(line: str) -> Dict:
        """
        解析一行 tegrastats 輸出
        
        Returns:
            dict 包含 ram_used_mb, ram_total_mb, swap_used_mb, cpu_pcts, gpu_pct, temps
        """
        result = {
            'ram_used_mb': 0,
            'ram_total_mb': 0,
            'swap_used_mb': 0,
            'cpu_pcts': [],
            'cpu_avg_pct': 0,
            'cpu_sum_pct': 0,
            'gpu_pct': 0,
            'temp_cpu': 0,
            'temp_gpu': 0,
        }
        
        try:
            # 解析 RAM: "RAM 5164/15523MB"
            ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
            if ram_match:
                result['ram_used_mb'] = int(ram_match.group(1))
                result['ram_total_mb'] = int(ram_match.group(2))
            
            # 解析 SWAP: "SWAP 0/7761MB"
            swap_match = re.search(r'SWAP (\d+)/(\d+)MB', line)
            if swap_match:
                result['swap_used_mb'] = int(swap_match.group(1))
            
            # 解析 CPU: "CPU [8%@1113,6%@1113,5%@1113,...]"
            cpu_match = re.search(r'CPU \[([^\]]+)\]', line)
            if cpu_match:
                cpu_str = cpu_match.group(1)
                # 提取每個核心的百分比，格式為 "8%@1113" 或 "off"
                cpu_parts = cpu_str.split(',')
                for part in cpu_parts:
                    pct_match = re.match(r'(\d+)%', part.strip())
                    if pct_match:
                        result['cpu_pcts'].append(int(pct_match.group(1)))
                
                if result['cpu_pcts']:
                    result['cpu_sum_pct'] = sum(result['cpu_pcts'])
                    result['cpu_avg_pct'] = result['cpu_sum_pct'] / len(result['cpu_pcts'])
            
            # 解析 GPU: "GR3D_FREQ 0%@[0]" 或 "GR3D_FREQ 50%"
            gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
            if gpu_match:
                result['gpu_pct'] = int(gpu_match.group(1))
            
            # 解析溫度: "CPU@55.812C" "GPU@53.281C"
            cpu_temp_match = re.search(r'CPU@([\d.]+)C', line)
            if cpu_temp_match:
                result['temp_cpu'] = float(cpu_temp_match.group(1))
                
            gpu_temp_match = re.search(r'GPU@([\d.]+)C', line)
            if gpu_temp_match:
                result['temp_gpu'] = float(gpu_temp_match.group(1))
                
        except Exception as e:
            rospy.logwarn(f"[TegrastatsParser] Parse error: {e}")
            
        return result


class ResourceMonitor:
    """
    資源監控類別 - 使用 tegrastats 進行系統監控
    """

    def __init__(self, log_dir: str = "logs", interval: float = 1.0):
        """
        初始化監控器

        Args:
            log_dir: Log 檔案存放目錄
            interval: 採樣間隔（秒）
        """
        self.log_dir = Path(log_dir)
        self.interval = interval
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._tegrastats_proc: Optional[subprocess.Popen] = None
        self.csv_path: Optional[Path] = None
        self.start_time: float = 0
        self.ready_time: float = 0
        self.baseline_ram_mb: float = 0  # 啟動前基準 RAM

        # 確保 Log 目錄存在
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """啟動監控"""
        if self.is_running:
            return

        # 建立新的 CSV 檔案
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"resources_{timestamp}.csv"

        rospy.loginfo(f"[ResourceMonitor] Starting monitoring to {self.csv_path}")
        
        # 記錄基準 RAM（啟動前）
        self._record_baseline_ram()

        self.is_running = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def _record_baseline_ram(self):
        """記錄啟動前的基準 RAM 使用量"""
        try:
            proc = subprocess.Popen(
                ['tegrastats', '--interval', '500'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            # 讀取一行來獲取基準值
            line = proc.stdout.readline()
            proc.terminate()
            proc.wait(timeout=2)
            
            stats = TegrastatsParser.parse_line(line)
            self.baseline_ram_mb = stats['ram_used_mb']
            rospy.loginfo(f"[ResourceMonitor] Baseline RAM: {self.baseline_ram_mb:.0f} MB")
        except Exception as e:
            rospy.logwarn(f"[ResourceMonitor] Failed to get baseline RAM: {e}")
            self.baseline_ram_mb = 0

    def mark_ready(self):
        """標記系統已就緒（用於計算啟動延遲）"""
        if self.start_time > 0 and self.ready_time == 0:
            self.ready_time = time.time()
            latency = self.ready_time - self.start_time
            rospy.loginfo(f"[ResourceMonitor] System Ready. Latency: {latency:.2f} s")

    def stop(self):
        """停止監控並產生報告"""
        if not self.is_running:
            return

        rospy.loginfo("[ResourceMonitor] Stopping monitoring...")
        self.is_running = False
        
        # 終止 tegrastats 進程
        if self._tegrastats_proc:
            self._tegrastats_proc.terminate()
            try:
                self._tegrastats_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._tegrastats_proc.kill()
        
        if self._thread:
            self._thread.join(timeout=2.0)

        # 產生統計報告
        self._generate_report()

    def _generate_report(self):
        """產生資源統計報告"""
        if not self.csv_path or not self.csv_path.exists():
            return

        rospy.loginfo("\n" + "=" * 40)
        rospy.loginfo("SYSTEM RESOURCE SUMMARY")
        rospy.loginfo("=" * 40)

        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                rospy.loginfo("No data collected.")
                return

            # 計算 Startup Latency
            if self.ready_time > 0 and self.start_time > 0:
                startup_latency = self.ready_time - self.start_time
                startup_latency_str = f"{startup_latency:.2f} s"
            else:
                startup_latency_str = "N/A"

            def safe_float(row, key):
                try:
                    return float(row.get(key, 0))
                except ValueError:
                    return 0.0

            # 提取數據
            cpu_sums = [safe_float(r, 'CPU_Sum_Pct') for r in rows]
            gpu_pcts = [safe_float(r, 'GPU_Pct') for r in rows]
            ram_used = [safe_float(r, 'RAM_Used_MB') for r in rows]
            app_ram_used = [safe_float(r, 'App_RAM_Used_MB') for r in rows]

            # 計算指標
            peak_cpu_sum = max(cpu_sums) if cpu_sums else 0
            avg_cpu_sum = sum(cpu_sums) / len(cpu_sums) if cpu_sums else 0
            peak_gpu = max(gpu_pcts) if gpu_pcts else 0
            avg_gpu = sum(gpu_pcts) / len(gpu_pcts) if gpu_pcts else 0
            peak_ram = max(ram_used) / 1024.0 if ram_used else 0
            peak_app_ram = max(app_ram_used) / 1024.0 if app_ram_used else 0
            avg_app_ram = (sum(app_ram_used) / len(app_ram_used)) / 1024.0 if app_ram_used else 0

            # 輸出報告
            rospy.loginfo(f"{'Metric':<25} {'Value'}")
            rospy.loginfo("-" * 40)
            rospy.loginfo(f"{'Startup latency':<25} {startup_latency_str}")
            rospy.loginfo(f"{'Baseline RAM':<25} {self.baseline_ram_mb / 1024:.2f} GB")
            rospy.loginfo(f"{'Peak System CPU':<25} {peak_cpu_sum:.0f}%")
            rospy.loginfo(f"{'Avg System CPU':<25} {avg_cpu_sum:.0f}%")
            rospy.loginfo(f"{'Peak System RAM':<25} {peak_ram:.2f} GB")
            rospy.loginfo(f"{'Peak App RAM (delta)':<25} {peak_app_ram:.2f} GB")
            rospy.loginfo(f"{'Avg App RAM (delta)':<25} {avg_app_ram:.2f} GB")
            rospy.loginfo(f"{'Peak GPU utilization':<25} {peak_gpu:.0f}%")
            rospy.loginfo(f"{'Avg GPU utilization':<25} {avg_gpu:.0f}%")
            rospy.loginfo("=" * 40 + "\n")

            # 另存報告
            report_path = self.csv_path.with_name(self.csv_path.stem + "_report.txt")
            with open(report_path, 'w') as f:
                f.write("SYSTEM RESOURCE SUMMARY (via tegrastats)\n")
                f.write("=" * 45 + "\n")
                f.write(f"Timestamp:              {datetime.now()}\n")
                f.write(f"Startup latency:        {startup_latency_str}\n")
                f.write(f"Baseline RAM:           {self.baseline_ram_mb / 1024:.2f} GB\n")
                f.write("-" * 45 + "\n")
                f.write(f"Peak System CPU:        {peak_cpu_sum:.0f}%\n")
                f.write(f"Avg System CPU:         {avg_cpu_sum:.0f}%\n")
                f.write(f"Peak System RAM:        {peak_ram:.2f} GB\n")
                f.write(f"Peak App RAM (delta):   {peak_app_ram:.2f} GB\n")
                f.write(f"Avg App RAM (delta):    {avg_app_ram:.2f} GB\n")
                f.write(f"Peak GPU utilization:   {peak_gpu:.0f}%\n")
                f.write(f"Avg GPU utilization:    {avg_gpu:.0f}%\n")
                f.write("=" * 45 + "\n")
                f.write(f"Raw data: {self.csv_path.name}\n")

            rospy.loginfo(f"[ResourceMonitor] Report saved to {report_path}")

        except Exception as e:
            rospy.logerr(f"[ResourceMonitor] Error generating report: {e}")

    def _monitor_loop(self):
        """監控迴圈 - 使用 tegrastats"""
        try:
            # 寫入 CSV Header
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    'Timestamp',
                    'Time_Rel_Sec',
                    'CPU_Avg_Pct',
                    'CPU_Sum_Pct',
                    'GPU_Pct',
                    'RAM_Used_MB',
                    'App_RAM_Used_MB',
                    'RAM_Total_MB',
                    'SWAP_Used_MB',
                    'Temp_GPU_C',
                    'Temp_CPU_C'
                ]
                writer.writerow(header)

            # 啟動 tegrastats 進程
            interval_ms = int(self.interval * 1000)
            self._tegrastats_proc = subprocess.Popen(
                ['tegrastats', '--interval', str(interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )

            rospy.loginfo(f"[ResourceMonitor] tegrastats started with interval {interval_ms}ms")

            while self.is_running and not rospy.is_shutdown():
                line = self._tegrastats_proc.stdout.readline()
                if not line:
                    break

                current_time = time.time()
                elapsed = current_time - self.start_time
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                # 解析 tegrastats 輸出
                stats = TegrastatsParser.parse_line(line)

                # 計算 App RAM（當前 RAM - 基準 RAM）
                app_ram_mb = max(0, stats['ram_used_mb'] - self.baseline_ram_mb)

                row = [
                    timestamp,
                    f"{elapsed:.2f}",
                    f"{stats['cpu_avg_pct']:.1f}",
                    f"{stats['cpu_sum_pct']:.0f}",
                    f"{stats['gpu_pct']:.0f}",
                    f"{stats['ram_used_mb']:.0f}",
                    f"{app_ram_mb:.0f}",
                    f"{stats['ram_total_mb']:.0f}",
                    f"{stats['swap_used_mb']:.0f}",
                    f"{stats['temp_gpu']:.1f}",
                    f"{stats['temp_cpu']:.1f}"
                ]

                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

        except Exception as e:
            rospy.logerr(f"[ResourceMonitor] Error in monitor loop: {e}")
            import traceback
            traceback.print_exc()


def main():
    """ROS 節點主函式"""
    rospy.init_node('resource_monitor_node', anonymous=False)

    # 從 ROS 參數伺服器讀取設定
    log_dir = rospy.get_param('~log_dir', 'logs')
    interval = rospy.get_param('~interval', 1.0)
    ready_delay = rospy.get_param('~ready_delay', 5.0)

    rospy.loginfo(f"[ResourceMonitor] Log dir: {log_dir}, Interval: {interval}s")

    monitor = ResourceMonitor(log_dir=log_dir, interval=interval)

    # 註冊 shutdown callback
    rospy.on_shutdown(monitor.stop)

    # 啟動監控
    monitor.start()

    # 等待其他節點啟動後標記系統就緒
    rospy.sleep(ready_delay)
    monitor.mark_ready()

    # 保持節點運行
    rospy.spin()


if __name__ == "__main__":
    main()
