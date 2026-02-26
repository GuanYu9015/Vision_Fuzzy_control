#!/usr/bin/python3
"""
BEV 轉換視覺化測試工具
BEV Transform Visualization Test Tool

功能：
1. 即時顯示相機影像與 BEV 鳥瞰圖
2. 驗證像素到地面座標的轉換
3. 標示網格線與停止距離
4. 支援滑鼠點擊顯示座標

使用方式：
    python3 test_bev_visualization.py [--camera 0] [--no-gui]
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import time

# 導入自定義模組
from mod_bev_transform import BEVTransformer


class BEVVisualizationTest:
    """BEV 視覺化測試工具"""
    
    def __init__(self, camera_id=0, camera_width=640, camera_height=480):
        """初始化"""
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # 初始化 BEV 轉換器
        self.bev = BEVTransformer()
        
        # 初始化相機
        self._init_camera(camera_id)
        
        # 滑鼠狀態
        self.mouse_x = 0
        self.mouse_y = 0
        self.clicked_points = []  # 儲存點擊的點
        
        # BEV 視窗大小
        self.bev_width = 400
        self.bev_height = 600
        
        print("BEV 視覺化測試工具已啟動")
        print("操作說明：")
        print("  滑鼠移動 - 顯示對應的地面座標")
        print("  左鍵點擊 - 標記點並保留")
        print("  右鍵點擊 - 清除所有標記點")
        print("  'g' 鍵 - 切換網格顯示")
        print("  's' 鍵 - 儲存當前畫面")
        print("  'q' 鍵 - 離開")
    
    def _init_camera(self, camera_id):
        """初始化相機"""
        # 嘗試 GStreamer
        pipeline = self._gstreamer_pipeline(camera_id)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("GStreamer 失敗，使用標準介面")
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟相機 {camera_id}")
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"相機已開啟: {actual_w}x{actual_h}")
    
    def _gstreamer_pipeline(self, sensor_id=0):
        """GStreamer pipeline"""
        return (
            f"v4l2src device=/dev/video{sensor_id} ! "
            f"video/x-raw, width=(int){self.camera_width}, height=(int){self.camera_height}, framerate=(fraction)30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        )
    
    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠回調函數"""
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左鍵：添加標記點
            if x < self.camera_width:  # 只在相機影像區域有效
                x_ground, y_ground = self.bev.pixel_to_ground(x, y)
                self.clicked_points.append({
                    'pixel': (x, y),
                    'ground': (x_ground, y_ground)
                })
                print(f"標記點: 像素({x}, {y}) -> 地面({x_ground:.3f}m, {y_ground:.3f}m)")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右鍵：清除所有標記
            self.clicked_points.clear()
            print("已清除所有標記點")
    
    def draw_camera_overlay(self, frame, show_grid=True):
        """在相機影像上繪製覆蓋層"""
        overlay = frame.copy()
        
        # 繪製網格（對應實際距離）
        if show_grid:
            # 計算幾個關鍵距離對應的 y 座標
            distances = [0.5, 1.0, 1.04, 1.5, 2.0, 2.5, 3.0]
            
            for dist in distances:
                # 找到對應這個距離的影像 y 座標（簡化：使用影像中心 x）
                # 這需要反向計算，這裡使用近似方法
                for v in range(self.camera_height - 1, -1, -1):
                    _, y_ground = self.bev.pixel_to_ground(self.camera_width // 2, v)
                    if y_ground >= dist:
                        color = (0, 0, 255) if dist == 1.04 else (100, 100, 100)
                        thickness = 2 if dist == 1.04 else 1
                        cv2.line(overlay, (0, v), (self.camera_width, v), color, thickness)
                        cv2.putText(overlay, f"{dist:.1f}m", (5, v - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        break
        
        # 繪製標記點
        for point in self.clicked_points:
            px, py = point['pixel']
            cv2.circle(overlay, (px, py), 8, (0, 255, 0), -1)
            cv2.circle(overlay, (px, py), 10, (0, 0, 0), 2)
        
        # 繪製滑鼠位置的座標資訊
        if 0 <= self.mouse_x < self.camera_width and 0 <= self.mouse_y < self.camera_height:
            x_ground, y_ground = self.bev.pixel_to_ground(self.mouse_x, self.mouse_y)
            
            # 計算誤差
            errors = self.bev.compute_errors(self.mouse_x, self.mouse_y)
            
            # 繪製十字準心
            cv2.drawMarker(overlay, (self.mouse_x, self.mouse_y), (255, 0, 255),
                          cv2.MARKER_CROSS, 20, 2)
            
            # 顯示座標資訊
            info_y = 30
            cv2.putText(overlay, f"Pixel: ({self.mouse_x}, {self.mouse_y})",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(overlay, f"Ground: ({x_ground:+.3f}m, {y_ground:.3f}m)",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(overlay, f"e_d: {errors['e_d']:.3f}m  e_l: {errors['e_l']:.3f}m",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return overlay
    
    def create_bev_view(self):
        """建立 BEV 視圖"""
        bev = np.zeros((self.bev_height, self.bev_width, 3), dtype=np.uint8)
        
        # 座標轉換參數
        pixels_per_meter = self.bev_height / (self.bev.max_distance_m + 0.5)
        
        def ground_to_bev(x_m, y_m):
            bev_x = int(self.bev_width / 2 + x_m * pixels_per_meter)
            bev_y = int(self.bev_height - (y_m + 0.2) * pixels_per_meter)
            return bev_x, bev_y
        
        # 繪製網格線
        for dist in np.arange(0.5, self.bev.max_distance_m + 0.5, 0.5):
            _, y = ground_to_bev(0, dist)
            if 0 <= y < self.bev_height:
                color = (0, 0, 255) if abs(dist - self.bev.stop_distance_m) < 0.01 else (50, 50, 50)
                thickness = 2 if abs(dist - self.bev.stop_distance_m) < 0.01 else 1
                cv2.line(bev, (0, y), (self.bev_width, y), color, thickness)
                cv2.putText(bev, f"{dist:.1f}m", (5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # 繪製中心線
        cv2.line(bev, (self.bev_width // 2, 0), (self.bev_width // 2, self.bev_height),
                (50, 50, 50), 1)
        
        # 繪製機器人
        robot_x, robot_y = ground_to_bev(0, 0)
        cv2.rectangle(bev, (robot_x - 15, robot_y - 8), (robot_x + 15, robot_y + 8),
                     (0, 255, 0), -1)
        
        # 繪製標記點
        for point in self.clicked_points:
            x_ground, y_ground = point['ground']
            bev_x, bev_y = ground_to_bev(x_ground, y_ground)
            if 0 <= bev_x < self.bev_width and 0 <= bev_y < self.bev_height:
                cv2.circle(bev, (bev_x, bev_y), 6, (0, 255, 0), -1)
        
        # 繪製當前滑鼠位置對應的點
        if 0 <= self.mouse_x < self.camera_width and 0 <= self.mouse_y < self.camera_height:
            x_ground, y_ground = self.bev.pixel_to_ground(self.mouse_x, self.mouse_y)
            bev_x, bev_y = ground_to_bev(x_ground, y_ground)
            if 0 <= bev_x < self.bev_width and 0 <= bev_y < self.bev_height:
                cv2.circle(bev, (bev_x, bev_y), 8, (255, 0, 255), -1)
                
                # 繪製從機器人到目標的線
                cv2.line(bev, (robot_x, robot_y), (bev_x, bev_y), (255, 255, 0), 2)
        
        # 標題
        cv2.putText(bev, "Bird's Eye View", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return bev
    
    def run(self, show_gui=True):
        """執行測試"""
        if not show_gui:
            self._run_headless_test()
            return
        
        cv2.namedWindow('BEV Visualization Test')
        cv2.setMouseCallback('BEV Visualization Test', self.mouse_callback)
        
        show_grid = True
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("無法讀取相機畫面")
                    continue
                
                # 繪製相機覆蓋層
                camera_view = self.draw_camera_overlay(frame, show_grid)
                
                # 建立 BEV 視圖
                bev_view = self.create_bev_view()
                
                # 調整 BEV 視圖高度以匹配相機影像
                bev_resized = cv2.resize(bev_view, (self.bev_width, self.camera_height))
                
                # 合併視圖
                combined = np.hstack([camera_view, bev_resized])
                
                cv2.imshow('BEV Visualization Test', combined)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('g'):
                    show_grid = not show_grid
                    print(f"網格顯示: {'開啟' if show_grid else '關閉'}")
                elif key == ord('s'):
                    filename = f"bev_test_{int(time.time())}.png"
                    cv2.imwrite(filename, combined)
                    print(f"已儲存: {filename}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def _run_headless_test(self):
        """無 GUI 測試"""
        print("\n===== BEV 轉換測試（無 GUI）=====\n")
        
        # 測試關鍵點
        test_points = [
            (320, 479),  # 底部中央
            (320, 400),
            (320, 300),
            (320, 200),
            (320, 100),
            (0, 300),    # 左邊
            (639, 300),  # 右邊
        ]
        
        print(f"{'像素 (u, v)':>15} | {'地面 (x, y)':>20} | {'e_d':>8} | {'e_l':>8}")
        print("-" * 60)
        
        for u, v in test_points:
            x_ground, y_ground = self.bev.pixel_to_ground(u, v)
            errors = self.bev.compute_errors(u, v)
            print(f"({u:3d}, {v:3d})      | ({x_ground:+.3f}m, {y_ground:.3f}m) | "
                  f"{errors['e_d']:+.3f}m | {errors['e_l']:+.3f}m")
        
        print("-" * 60)
        
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description='BEV 轉換視覺化測試')
    parser.add_argument('--camera', type=int, default=0, help='相機 ID')
    parser.add_argument('--width', type=int, default=640, help='影像寬度')
    parser.add_argument('--height', type=int, default=480, help='影像高度')
    parser.add_argument('--no-gui', action='store_true', help='無 GUI 模式')
    
    args = parser.parse_args()
    
    tester = BEVVisualizationTest(
        camera_id=args.camera,
        camera_width=args.width,
        camera_height=args.height
    )
    
    tester.run(show_gui=not args.no_gui)


if __name__ == "__main__":
    main()
