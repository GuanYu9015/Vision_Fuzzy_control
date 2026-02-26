#!/usr/bin/python3
"""
相機標定工具
Camera Calibration Tool using Chessboard Pattern

使用方式：
1. 列印棋盤格圖案 (預設 9x6 內角點)
2. 執行此腳本
3. 將棋盤格放置在相機前方不同位置與角度，按 's' 鍵擷取
4. 擷取 15-20 張後，按 'c' 鍵執行標定
5. 標定結果會自動儲存到 camera_config.yaml

快捷鍵：
  s - 擷取當前幀用於標定
  c - 執行標定計算
  u - 顯示去畸變後的影像
  q - 離開
"""

import os
import sys
import cv2
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime


class CameraCalibrator:
    """相機標定器"""
    
    def __init__(self, 
                 chessboard_size=(9, 6),
                 square_size_mm=25.0,
                 camera_id=0,
                 camera_width=640,
                 camera_height=480,
                 config_path=None):
        """
        初始化標定器
        
        Args:
            chessboard_size: 棋盤格內角點數量 (列, 行)
            square_size_mm: 每個方格的邊長 (毫米)
            camera_id: 相機設備 ID
            camera_width: 影像寬度
            camera_height: 影像高度
            config_path: 配置檔案路徑
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size_mm / 1000.0  # 轉換為公尺
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # 設定配置檔案路徑
        if config_path is None:
            script_dir = Path(__file__).parent.parent
            config_path = script_dir / 'config' / 'camera_config.yaml'
        self.config_path = Path(config_path)
        
        # 準備物體點 (棋盤格在世界座標系中的位置)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        # 儲存標定用的點
        self.obj_points = []  # 3D 世界座標
        self.img_points = []  # 2D 影像座標
        
        # 標定結果
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibrated = False
        
        # 初始化相機
        self._init_camera(camera_id)
        
        print(f"相機標定工具已初始化")
        print(f"棋盤格大小: {chessboard_size[0]}x{chessboard_size[1]} 內角點")
        print(f"方格邊長: {square_size_mm} mm")
        print(f"配置檔案: {self.config_path}")
    
    def _init_camera(self, camera_id):
        """初始化相機"""
        # 嘗試使用 GStreamer (Jetson)
        pipeline = self._gstreamer_pipeline(camera_id)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            # 回退到普通相機
            print("GStreamer 初始化失敗，使用標準相機介面")
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟相機 {camera_id}")
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"相機已開啟: {actual_w}x{actual_h}")
    
    def _gstreamer_pipeline(self, sensor_id=0):
        """建立 GStreamer pipeline"""
        return (
            f"v4l2src device=/dev/video{sensor_id} ! "
            f"video/x-raw, width=(int){self.camera_width}, height=(int){self.camera_height}, framerate=(fraction)30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        )
    
    def find_chessboard(self, frame):
        """
        在影像中尋找棋盤格
        
        Returns:
            found: 是否找到
            corners: 角點座標 (如果找到)
            vis_frame: 視覺化結果
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 尋找棋盤格角點
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, flags)
        
        vis_frame = frame.copy()
        
        if found:
            # 亞像素級精確化角點位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 繪製角點
            cv2.drawChessboardCorners(vis_frame, self.chessboard_size, corners, found)
            
            # 顯示狀態
            cv2.putText(vis_frame, "Chessboard FOUND - Press 's' to capture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis_frame, "Chessboard NOT found", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 顯示已擷取的幀數
        cv2.putText(vis_frame, f"Captured: {len(self.img_points)} frames", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return found, corners, vis_frame
    
    def add_calibration_frame(self, corners):
        """新增一幀標定資料"""
        self.obj_points.append(self.objp)
        self.img_points.append(corners)
        print(f"已擷取第 {len(self.img_points)} 幀標定資料")
    
    def calibrate(self):
        """執行相機標定"""
        if len(self.img_points) < 10:
            print(f"警告：建議至少擷取 10 幀資料，目前只有 {len(self.img_points)} 幀")
            if len(self.img_points) < 5:
                print("錯誤：至少需要 5 幀資料才能進行標定")
                return False
        
        print(f"\n開始標定，使用 {len(self.img_points)} 幀資料...")
        
        # 執行標定
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.obj_points, 
            self.img_points, 
            (self.camera_width, self.camera_height),
            None, 
            None
        )
        
        if ret:
            self.calibrated = True
            print("\n===== 標定結果 =====")
            print(f"重投影誤差 (RMS): {ret:.4f} 像素")
            print(f"\n相機內參矩陣 K:")
            print(self.camera_matrix)
            print(f"\n畸變係數 (k1, k2, p1, p2, k3):")
            print(self.dist_coeffs.flatten())
            print(f"\n焦距: fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
            print(f"主點: cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}")
            
            # 儲存結果
            self._save_calibration()
            return True
        else:
            print("標定失敗")
            return False
    
    def _save_calibration(self):
        """儲存標定結果到配置檔案"""
        if not self.calibrated:
            print("尚未完成標定，無法儲存")
            return
        
        # 讀取現有配置
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # 更新相機內參
        if 'camera_intrinsics' not in config:
            config['camera_intrinsics'] = {}
        
        config['camera_intrinsics']['width'] = self.camera_width
        config['camera_intrinsics']['height'] = self.camera_height
        config['camera_intrinsics']['fx'] = float(self.camera_matrix[0, 0])
        config['camera_intrinsics']['fy'] = float(self.camera_matrix[1, 1])
        config['camera_intrinsics']['cx'] = float(self.camera_matrix[0, 2])
        config['camera_intrinsics']['cy'] = float(self.camera_matrix[1, 2])
        config['camera_intrinsics']['distortion'] = self.dist_coeffs.flatten().tolist()
        config['camera_intrinsics']['calibrated'] = True
        
        # 儲存
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n標定結果已儲存到: {self.config_path}")
        
        # 同時儲存備份 (含時間戳)
        backup_dir = self.config_path.parent / 'calibration_backup'
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"calibration_{timestamp}.yaml"
        
        calib_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'image_size': [self.camera_width, self.camera_height],
            'calibration_frames': len(self.img_points),
            'timestamp': timestamp
        }
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(calib_data, f, default_flow_style=False)
        
        print(f"備份已儲存到: {backup_path}")
    
    def undistort_frame(self, frame):
        """去除影像畸變"""
        if not self.calibrated:
            return frame
        
        h, w = frame.shape[:2]
        
        # 計算最佳新相機矩陣
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # 去畸變
        undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        # 裁剪 ROI
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def run(self):
        """執行標定流程"""
        print("\n===== 相機標定工具 =====")
        print("操作說明：")
        print("  s - 擷取當前幀 (需要先偵測到棋盤格)")
        print("  c - 執行標定計算 (需要至少 10 幀)")
        print("  u - 切換顯示去畸變影像")
        print("  q - 離開")
        print("========================\n")
        
        show_undistorted = False
        current_corners = None
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("無法讀取相機畫面")
                    continue
                
                # 尋找棋盤格
                found, corners, vis_frame = self.find_chessboard(frame)
                if found:
                    current_corners = corners
                else:
                    current_corners = None
                
                # 顯示去畸變影像
                if show_undistorted and self.calibrated:
                    vis_frame = self.undistort_frame(vis_frame)
                    cv2.putText(vis_frame, "UNDISTORTED", 
                               (10, vis_frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                cv2.imshow('Camera Calibration', vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s') and current_corners is not None:
                    self.add_calibration_frame(current_corners)
                elif key == ord('c'):
                    self.calibrate()
                elif key == ord('u'):
                    show_undistorted = not show_undistorted
                    print(f"去畸變顯示: {'開啟' if show_undistorted else '關閉'}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description='相機標定工具')
    parser.add_argument('--rows', type=int, default=6, help='棋盤格內角點行數')
    parser.add_argument('--cols', type=int, default=9, help='棋盤格內角點列數')
    parser.add_argument('--square-size', type=float, default=25.0, help='方格邊長 (mm)')
    parser.add_argument('--camera', type=int, default=0, help='相機 ID')
    parser.add_argument('--width', type=int, default=640, help='影像寬度')
    parser.add_argument('--height', type=int, default=480, help='影像高度')
    parser.add_argument('--config', type=str, default=None, help='配置檔案路徑')
    
    args = parser.parse_args()
    
    calibrator = CameraCalibrator(
        chessboard_size=(args.cols, args.rows),
        square_size_mm=args.square_size,
        camera_id=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        config_path=args.config
    )
    
    calibrator.run()


if __name__ == "__main__":
    main()
