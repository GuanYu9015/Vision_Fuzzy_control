#!/usr/bin/python3
"""
鳥瞰圖座標轉換模組
Bird's Eye View (BEV) Transform Module

功能：
1. 將相機影像中的像素座標轉換為地面實際座標（公尺）
2. 計算模糊控制器所需的誤差值
3. 支援透視變換視覺化

原理：
相機以俯仰角 θ 安裝，透過射線投影計算地面交點
"""

import numpy as np
import yaml
import cv2
from pathlib import Path
from typing import Tuple, Dict, Optional
import time


class BEVTransformer:
    """鳥瞰圖座標轉換器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化轉換器
        
        Args:
            config_path: 相機配置檔案路徑
        """
        # 設定配置檔案路徑
        if config_path is None:
            # 嘗試多個可能的路徑
            script_dir = Path(__file__).parent
            possible_paths = [
                script_dir.parent / 'config' / 'camera_config.yaml',  # ../config/
                script_dir / 'config' / 'camera_config.yaml',         # ./config/
                Path('/home/cir/ros/src/yolov7_ros/config/camera_config.yaml'),  # 絕對路徑
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                config_path = possible_paths[0]  # 使用第一個作為預設（用於錯誤訊息）
        
        self.config_path = Path(config_path)
        self._load_config()
        
        # 計算旋轉矩陣（俯仰角）
        self._compute_rotation_matrix()
        
        # 誤差計算用的狀態變數
        self.prev_e_d = None  # 前一時刻的前方距離誤差
        self.prev_e_l = None  # 前一時刻的橫向誤差
        self.prev_time = None  # 前一時刻的時間戳
        
        print(f"BEV 轉換器已初始化")
        print(f"相機高度: {self.camera_height_m:.3f} m")
        print(f"俯仰角: {self.pitch_deg:.1f}°")
        print(f"停止距離: {self.stop_distance_m:.2f} m")
    
    def _load_config(self):
        """載入配置檔案"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"找不到配置檔案: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 相機內參
        intrinsics = config['camera_intrinsics']
        self.image_width = intrinsics['width']
        self.image_height = intrinsics['height']
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        self.distortion = np.array(intrinsics.get('distortion', [0, 0, 0, 0, 0]))
        
        # 建構內參矩陣
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        # 相機外參
        extrinsics = config['camera_extrinsics']
        self.pitch_deg = extrinsics['pitch_deg']
        self.pitch_rad = np.radians(self.pitch_deg)
        self.camera_height_m = extrinsics['height_m']
        self.offset_from_robot_front = extrinsics.get('offset_from_robot_front_m', 0)
        
        # 安全參數
        safety = config['safety']
        self.stop_distance_m = safety['stop_distance_m']
        
        # 感測範圍
        sensing = config['sensing_range']
        self.min_distance_m = sensing.get('min_distance_from_camera_m', 0.45)
        self.max_distance_m = sensing.get('max_distance_from_camera_m', 3.25)
        
        # 模糊控制範圍
        if 'fuzzy_ranges' in config:
            fuzzy = config['fuzzy_ranges']
            self.e_d_range = (fuzzy['e_d_min'], fuzzy['e_d_max'])
            self.e_l_range = (fuzzy['e_l_min'], fuzzy['e_l_max'])
        else:
            self.e_d_range = (0, 2.08)
            self.e_l_range = (-0.5, 0.5)
    
    def _compute_rotation_matrix(self):
        """預計算三角函數值和校正係數"""
        # 俯仰角：向下為正
        # 例如 29° 表示相機光軸向下傾斜 29°
        self.cos_pitch = np.cos(self.pitch_rad)
        self.sin_pitch = np.sin(self.pitch_rad)
        self.tan_pitch = np.tan(self.pitch_rad)
        
        # 垂直距離校正係數（基於實測數據的二次多項式擬合）
        # 將計算值轉換為實際距離: actual = a*d^2 + b*d + c
        # 擬合數據:
        #   顯示值: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] m
        #   實際值: [0.59, 1.06, 1.46, 1.81, 2.10, 2.40] m
        self.distance_correction_coeffs = np.array([-0.0921, 1.0379, 0.1030])
        
        # 水平距離校正係數（與前方距離 y 相關的線性校正）
        # 校正因子 = a * y + b
        # 擬合數據:
        #   y=1.0m 時校正因子約 0.844
        #   y=2.0m 時校正因子約 0.682
        self.lateral_correction_a = -0.1613   # 斜率
        self.lateral_correction_b = 1.0051    # 截距
        
        # 相機水平偏移 - 目前設為 0
        self.camera_lateral_offset_m = 0.0
    
    def _correct_distance(self, raw_distance: float) -> float:
        """
        應用垂直距離校正
        
        Args:
            raw_distance: 原始計算的距離 (公尺)
            
        Returns:
            corrected_distance: 校正後的距離 (公尺)
        """
        # 多項式校正: actual = a*d^2 + b*d + c
        corrected = np.polyval(self.distance_correction_coeffs, raw_distance)
        return float(corrected)
    
    def _correct_lateral(self, raw_lateral: float, y_distance: float) -> float:
        """
        應用水平距離校正（與前方距離相關）
        
        Args:
            raw_lateral: 原始計算的橫向距離 (公尺)
            y_distance: 前方距離 (公尺)
            
        Returns:
            corrected_lateral: 校正後的橫向距離 (公尺)
        """
        # 線性校正因子 = a * y + b
        factor = self.lateral_correction_a * y_distance + self.lateral_correction_b
        corrected = raw_lateral * factor + self.camera_lateral_offset_m
        return float(corrected)
    
    def pixel_to_normalized(self, u: float, v: float) -> Tuple[float, float]:
        """
        像素座標轉換為歸一化相機座標
        
        Args:
            u: 像素 x 座標
            v: 像素 y 座標
            
        Returns:
            (x_norm, y_norm): 歸一化座標
        """
        x_norm = (u - self.cx) / self.fx
        y_norm = (v - self.cy) / self.fy
        return x_norm, y_norm
    
    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """
        將像素座標轉換為地面座標（公尺）
        
        使用幾何投影法：
        - 相機在高度 h 處，光軸向下傾斜 θ 度
        - 影像中心點 (cx, cy) 對應光軸方向
        - 光軸與地面的交點距離相機水平距離 = h / tan(θ)
        
        Args:
            u: 像素 x 座標 (0~width-1)
            v: 像素 y 座標 (0~height-1)
            
        Returns:
            (x_ground, y_ground): 地面座標（公尺）
            - x_ground: 橫向偏移（左負右正）
            - y_ground: 前方距離（相機到目標點的水平距離）
        """
        # 步驟 1: 計算像素相對於主點的角度偏移
        # 水平角度偏移 (正值 = 向右)
        delta_u = u - self.cx
        alpha = np.arctan2(delta_u, self.fx)  # 水平偏角
        
        # 垂直角度偏移 (正值 = 向下，因為影像 y 軸向下)
        delta_v = v - self.cy
        beta = np.arctan2(delta_v, self.fy)  # 垂直偏角
        
        # 步驟 2: 計算射線與地面的實際俯仰角
        # 總俯仰角 = 相機俯仰角 + 像素垂直偏角
        # 正值表示向下看
        total_pitch = self.pitch_rad + beta
        
        # 步驟 3: 計算前方距離 (Y)
        # 如果俯仰角 <= 0，射線向上或水平，不會與地面相交
        if total_pitch <= 0:
            return 0.0, self.max_distance_m
        
        # 前方距離 = 相機高度 / tan(總俯仰角)
        y_raw = self.camera_height_m / np.tan(total_pitch)
        
        # 步驟 4: 應用垂直距離校正
        y_ground = self._correct_distance(y_raw)
        
        # 步驟 5: 計算橫向距離 (X)
        # 考慮到相機傾斜，需要用斜距計算
        # 斜距 = 相機高度 / sin(總俯仰角)
        slant_distance = self.camera_height_m / np.sin(total_pitch)
        x_raw = slant_distance * np.tan(alpha)
        
        # 步驟 6: 應用水平距離校正（與前方距離相關）
        x_ground = self._correct_lateral(x_raw, y_ground)
        
        # 限制在有效範圍內
        y_ground = np.clip(y_ground, self.min_distance_m, self.max_distance_m)
        
        return float(x_ground), float(y_ground)
    
    def distance_to_pixel_y(self, distance_m: float) -> int:
        """
        從地面距離（公尺）計算對應的像素 y 座標（逆計算）
        
        Args:
            distance_m: 前方距離（公尺）
            
        Returns:
            pixel_y: 對應的像素 y 座標
        """
        # 逆向應用距離校正（近似）
        # 使用二分搜尋找到對應的原始距離
        # 校正公式: actual = a*d^2 + b*d + c
        # 需要反解 d
        a, b, c = self.distance_correction_coeffs
        
        # 解二次方程: a*d^2 + b*d + (c - distance_m) = 0
        discriminant = b**2 - 4*a*(c - distance_m)
        if discriminant < 0:
            y_raw = distance_m  # 若無解，使用原值
        else:
            # 取正根
            y_raw = (-b + np.sqrt(discriminant)) / (2*a)
            if y_raw < 0:
                y_raw = (-b - np.sqrt(discriminant)) / (2*a)
        
        # 從 y_raw 計算總俯仰角
        # y_raw = h / tan(total_pitch)
        # total_pitch = arctan(h / y_raw)
        if y_raw <= 0:
            return int(self.cy)
        
        total_pitch = np.arctan(self.camera_height_m / y_raw)
        
        # 計算像素垂直偏角
        # total_pitch = pitch_rad + beta
        beta = total_pitch - self.pitch_rad
        
        # 從偏角計算像素座標
        # beta = arctan2(delta_v, fy)
        # delta_v = fy * tan(beta)
        delta_v = self.fy * np.tan(beta)
        
        # v = cy + delta_v
        pixel_y = int(self.cy + delta_v)
        
        # 限制在有效範圍內
        pixel_y = np.clip(pixel_y, 0, self.image_height - 1)
        
        return int(pixel_y)
    
    def compute_errors(self, 
                       forward_ref_x: float, 
                       forward_ref_y: float,
                       lateral_ref_x: float,
                       lateral_ref_y: float,
                       timestamp: Optional[float] = None) -> Dict[str, float]:
        """
        計算模糊控制器所需的誤差值（分離參考點）
        
        Args:
            forward_ref_x: 前方距離參考點像素 x 座標
            forward_ref_y: 前方距離參考點像素 y 座標
            lateral_ref_x: 橫向誤差參考點像素 x 座標
            lateral_ref_y: 橫向誤差參考點像素 y 座標
            timestamp: 時間戳（秒），用於計算變化率
            
        Returns:
            dict: 包含以下鍵值的字典
                - e_d: 前方距離誤差 (公尺)
                - e_d_dot: 前方距離誤差變化率 (m/s)
                - e_l: 橫向誤差 (公尺)
                - e_l_dot: 橫向誤差變化率 (m/s)
                - x_ground: 橫向參考點地面 x 座標
                - y_ground: 前方參考點地面 y 座標 (前方距離)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 前方距離使用 forward_ref
        _, y_ground = self.pixel_to_ground(forward_ref_x, forward_ref_y)
        
        # 橫向誤差使用 lateral_ref
        x_ground, _ = self.pixel_to_ground(lateral_ref_x, lateral_ref_y)
        
        # 計算前方距離誤差
        # e_d = 當前距離 - 停止距離
        # 正值表示還有距離可以前進
        e_d = y_ground - self.stop_distance_m
        e_d = np.clip(e_d, self.e_d_range[0], self.e_d_range[1])
        
        # 計算橫向誤差
        # e_l = x_ground (已經是相對於中心的偏移)
        e_l = x_ground
        e_l = np.clip(e_l, self.e_l_range[0], self.e_l_range[1])
        
        # 計算變化率
        e_d_dot = 0.0
        e_l_dot = 0.0
        
        if self.prev_time is not None and self.prev_e_d is not None:
            dt = timestamp - self.prev_time
            if dt > 0.001:  # 避免除以零
                e_d_dot = (e_d - self.prev_e_d) / dt
                e_l_dot = (e_l - self.prev_e_l) / dt
                
                # 限制變化率範圍
                e_d_dot = np.clip(e_d_dot, -2.08, 2.08)
                e_l_dot = np.clip(e_l_dot, -1.0, 1.0)
        
        # 更新狀態
        self.prev_e_d = e_d
        self.prev_e_l = e_l
        self.prev_time = timestamp
        
        return {
            'e_d': float(e_d),
            'e_d_dot': float(e_d_dot),
            'e_l': float(e_l),
            'e_l_dot': float(e_l_dot),
            'x_ground': float(x_ground),
            'y_ground': float(y_ground)
        }
    
    def reset_state(self):
        """重置狀態變數（用於新的追蹤週期）"""
        self.prev_e_d = None
        self.prev_e_l = None
        self.prev_time = None
    
    def create_bev_visualization(self, 
                                  frame: np.ndarray,
                                  forward_ref_x: Optional[float] = None,
                                  forward_ref_y: Optional[float] = None,
                                  lateral_ref_x: Optional[float] = None,
                                  lateral_ref_y: Optional[float] = None,
                                  bev_size: Tuple[int, int] = (400, 480)) -> np.ndarray:
        """
        建立鳥瞰圖視覺化（支援雙參考點）
        
        Args:
            frame: 原始相機影像
            forward_ref_x: 前方距離參考點像素 x 座標（用於 e_d）
            forward_ref_y: 前方距離參考點像素 y 座標
            lateral_ref_x: 橫向誤差參考點像素 x 座標（用於 e_l）
            lateral_ref_y: 橫向誤差參考點像素 y 座標
            bev_size: BEV 輸出大小 (寬, 高)
            
        Returns:
            bev_image: 鳥瞰圖影像
        """
        bev_w, bev_h = bev_size
        
        # 建立 BEV 影像（黑底）
        bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
        
        # BEV 座標系：
        # - 原點在底部中央（機器人位置）
        # - Y 軸向上（前方）
        # - X 軸向右
        
        # 計算顯示範圍
        # Y 軸（前方）：0 ~ max_distance_m
        # X 軸（橫向）：使用與 Y 軸相同的比例尺
        y_range = self.max_distance_m + 0.5  # 前方顯示範圍 (公尺)
        x_range = y_range * bev_w / bev_h    # 根據視窗比例計算橫向範圍
        
        # 公尺到像素的比例（X 和 Y 使用相同比例尺）
        pixels_per_meter = bev_h / y_range
        
        def ground_to_bev(x_m, y_m):
            """地面座標轉 BEV 像素座標"""
            bev_x = int(bev_w / 2 + x_m * pixels_per_meter)
            # y_m = 0 時在底部，y_m 增加時向上移動
            bev_y = int(bev_h - (y_m + 0.2) * pixels_per_meter)
            return bev_x, bev_y
        
        # ===== 繪製模糊隸屬函數範圍 =====
        # e_d 範圍區域（水平色帶）- 基於停止距離計算
        # e_d = y_ground - stop_distance，所以 y_ground = e_d + stop_distance
        e_d_zones = [
            # (e_d_min, e_d_max, label, color BGR)
            (0.0, 0.52, 'VN', (0, 0, 150)),      # 深紅 - Very Near (危險)
            (0.52, 1.04, 'N', (0, 100, 200)),    # 橙色 - Near
            (1.04, 1.56, 'M', (0, 150, 150)),    # 黃色 - Medium
            (1.56, 2.08, 'F', (0, 150, 0)),      # 綠色 - Far (安全)
        ]
        
        for e_d_min, e_d_max, label, color in e_d_zones:
            # 轉換 e_d 到實際 y 座標（y = e_d + stop_distance）
            y_min = e_d_min + self.stop_distance_m
            y_max = e_d_max + self.stop_distance_m
            
            _, bev_y_max = ground_to_bev(0, y_min)  # e_d 小 -> 靠近機器人 -> bev_y 大
            _, bev_y_min = ground_to_bev(0, y_max)  # e_d 大 -> 遠離機器人 -> bev_y 小
            
            # 限制在視窗範圍內
            bev_y_min = max(0, bev_y_min)
            bev_y_max = min(bev_h - 1, bev_y_max)
            
            if bev_y_min < bev_y_max:
                # 繪製半透明色帶（左右邊緣）
                cv2.rectangle(bev, (0, bev_y_min), (25, bev_y_max), color, -1)
                cv2.rectangle(bev, (bev_w - 25, bev_y_min), (bev_w, bev_y_max), color, -1)
                # 標籤
                label_y = (bev_y_min + bev_y_max) // 2
                cv2.putText(bev, label, (5, label_y + 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # e_l 範圍區域（垂直線）- 橫向誤差邊界
        e_l_boundaries = [
            # (e_l_value, label, color BGR)
            (-0.5, 'NB', (200, 100, 0)),    # 藍色 - 左偏大邊界
            (-0.20, 'NS', (150, 150, 0)),   # 青色 - 與模糊控制器一致
            (0.20, 'PS', (0, 150, 150)),    # 黃色 - 與模糊控制器一致
            (0.5, 'PB', (0, 100, 200)),     # 橙色 - 右偏大邊界
        ]
        
        for e_l_val, label, color in e_l_boundaries:
            bev_x, _ = ground_to_bev(e_l_val, 0)
            if 0 < bev_x < bev_w:
                cv2.line(bev, (bev_x, 0), (bev_x, bev_h), color, 1, cv2.LINE_AA)
        
        # 繪製 ZO 區域（中心安全區，半透明綠色）- 縮小死區
        zo_left, _ = ground_to_bev(-0.10, 0)
        zo_right, _ = ground_to_bev(0.10, 0)
        overlay = bev.copy()
        cv2.rectangle(overlay, (zo_left, 0), (zo_right, bev_h), (0, 80, 0), -1)
        cv2.addWeighted(overlay, 0.3, bev, 0.7, 0, bev)
        
        # ===== 繪製網格線（每 0.5 公尺）=====
        for dist in np.arange(0.5, self.max_distance_m + 0.5, 0.5):
            _, y = ground_to_bev(0, dist)
            if 0 <= y < bev_h:
                cv2.line(bev, (0, y), (bev_w, y), (50, 50, 50), 1)
                cv2.putText(bev, f"{dist:.1f}m", (30, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # 繪製中心線（垂直）
        cv2.line(bev, (bev_w // 2, 0), (bev_w // 2, bev_h), (50, 50, 50), 1)
        
        # 繪製停止線（紅色）
        _, stop_y = ground_to_bev(0, self.stop_distance_m)
        cv2.line(bev, (0, stop_y), (bev_w, stop_y), (0, 0, 255), 2)
        cv2.putText(bev, f"STOP {self.stop_distance_m:.2f}m", (30, stop_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 繪製機器人位置（底部中央，與 test_bev_visualization.py 相同尺寸）
        robot_x, robot_y = ground_to_bev(0, 0)
        cv2.rectangle(bev, 
                     (robot_x - 15, robot_y - 8),
                     (robot_x + 15, robot_y + 8),
                     (0, 255, 0), -1)
        
        # 標題
        cv2.putText(bev, "Bird's Eye View", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y = 50  # 資訊顯示起始 y 座標
        
        # 繪製前方距離參考點（藍色）- 用於 e_d
        if forward_ref_x is not None and forward_ref_y is not None:
            x_fwd, y_fwd = self.pixel_to_ground(forward_ref_x, forward_ref_y)
            fwd_bev_x, fwd_bev_y = ground_to_bev(x_fwd, y_fwd)
            
            # 繪製藍色圓點
            cv2.circle(bev, (fwd_bev_x, fwd_bev_y), 6, (255, 0, 0), -1)
            
            # 計算 e_d（前方距離誤差）
            e_d = y_fwd - self.stop_distance_m
            
            # 顯示前方參考點資訊
            cv2.putText(bev, f"Forward (e_d):", (bev_w - 140, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
            info_y += 18
            cv2.putText(bev, f"  Y: {y_fwd:.2f}m", (bev_w - 140, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += 16
            cv2.putText(bev, f"  e_d: {e_d:.2f}m", (bev_w - 140, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += 20
        
        # 繪製橫向誤差參考點（黃色）- 用於 e_l
        if lateral_ref_x is not None and lateral_ref_y is not None:
            x_lat, y_lat = self.pixel_to_ground(lateral_ref_x, lateral_ref_y)
            lat_bev_x, lat_bev_y = ground_to_bev(x_lat, y_lat)
            
            # 繪製黃色圓點
            cv2.circle(bev, (lat_bev_x, lat_bev_y), 6, (0, 255, 255), -1)
            
            # 顯示橫向參考點資訊
            cv2.putText(bev, f"Lateral (e_l):", (bev_w - 140, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            info_y += 18
            cv2.putText(bev, f"  X: {x_lat:+.2f}m", (bev_w - 140, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += 16
            cv2.putText(bev, f"  e_l: {x_lat:+.2f}m", (bev_w - 140, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return bev
    
    def get_pixel_grid_to_ground(self, 
                                  grid_step: int = 40) -> np.ndarray:
        """
        計算像素網格對應的地面座標
        
        Args:
            grid_step: 網格步長 (像素)
            
        Returns:
            grid_data: 形狀為 (n_points, 4) 的陣列
                       每列為 [u, v, x_ground, y_ground]
        """
        points = []
        
        for v in range(0, self.image_height, grid_step):
            for u in range(0, self.image_width, grid_step):
                x_ground, y_ground = self.pixel_to_ground(u, v)
                points.append([u, v, x_ground, y_ground])
        
        return np.array(points)


def test_bev_transform():
    """測試 BEV 轉換"""
    print("\n===== BEV 轉換測試 =====\n")
    
    try:
        transformer = BEVTransformer()
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        return
    
    # 測試幾個關鍵點
    test_points = [
        (320, 479, "影像底部中央（最近）"),
        (320, 240, "影像中央"),
        (320, 100, "影像頂部附近（較遠）"),
        (0, 240, "左邊緣"),
        (639, 240, "右邊緣"),
    ]
    
    print("像素到地面座標轉換測試：")
    print("-" * 60)
    
    for u, v, desc in test_points:
        x_ground, y_ground = transformer.pixel_to_ground(u, v)
        print(f"像素 ({u:3d}, {v:3d}) -> 地面 (x={x_ground:+.3f}m, y={y_ground:.3f}m) [{desc}]")
    
    print("-" * 60)
    
    # 測試誤差計算（雙參考點）
    print("\n誤差計算測試：")
    # forward_ref: (320, 200), lateral_ref: (320, 240)
    errors = transformer.compute_errors(320, 200, 320, 240)
    print(f"前方參考點 (320, 200), 橫向參考點 (320, 240):")
    print(f"  前方距離誤差 e_d: {errors['e_d']:.3f} m")
    print(f"  橫向誤差 e_l: {errors['e_l']:.3f} m")
    print(f"  地面座標: x={errors['x_ground']:.3f}m, y={errors['y_ground']:.3f}m")


if __name__ == "__main__":
    test_bev_transform()
