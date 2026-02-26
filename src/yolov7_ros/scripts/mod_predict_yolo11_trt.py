#!/usr/bin/python3
"""
YOLO11n-seg TensorRT 道路檢測節點
基於 Ultralytics YOLO11 和 TensorRT 引擎進行實例分割
用於替換原本的 YOLOv7-seg 道路檢測模組

功能：
1. 使用 YOLO11n-seg TensorRT 引擎進行道路分割
2. 後處理找出道路參考點
3. 發布道路資訊到 /road_info topic
4. 支援 ROS launch 參數配置
5. 只顯示最高置信度的辨識物件 (max_det=1)
6. 儲存辨識結果影片 (MP4 格式，VLC 相容)
7. 儲存遮罩圖 (mask)
"""

import os
import sys
import signal
import time
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import rospy
import rospkg
from std_msgs.msg import Float32MultiArray, String

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 導入 Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"錯誤：無法導入 ultralytics 套件，請執行: pip install ultralytics")
    sys.exit(1)

# 導入 BEV 轉換模組
try:
    from mod_bev_transform import BEVTransformer
except ImportError:
    # 如果直接導入失敗，嘗試從同目錄導入
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from mod_bev_transform import BEVTransformer


class GracefulKiller:
    """優雅退出處理器"""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        rospy.loginfo("接收到終止信號，準備退出...")



def filter_ros_args():
    """過濾掉ROS相關的參數，只保留腳本自身的參數"""
    filtered_args = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i].startswith('__'):
            i += 1
            if i < len(sys.argv) and not sys.argv[i].startswith('-'):
                i += 1
            continue
        filtered_args.append(sys.argv[i])
        i += 1
    return filtered_args


# 在解析參數之前過濾掉ROS參數
sys.argv = filter_ros_args()


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    """
    建立 GStreamer Pipeline 字串
    在 Jetson 上使用 GStreamer 可以確保相機以正確的解析度和幀率運行
    """
    return (
        "v4l2src device=/dev/video%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
        )
    )


class YOLO11RoadDetector:
    """YOLO11 道路檢測器類別"""
    
    def __init__(self, weights_path, device='0', conf_thres=0.5, iou_thres=0.5, 
                 imgsz=640, max_det=1, camera_id=0, camera_width=640, camera_height=480,
                 view_img=True, use_gstreamer=True, save_video=False, save_mask=False,
                 output_dir='/home/cir/ros/src/yolov7_ros/src/yolo11/outputs'):
        """
        初始化 YOLO11 道路檢測器
        
        Args:
            weights_path: TensorRT 引擎檔案路徑 (.engine)
            device: CUDA 設備 ID
            conf_thres: 置信度閾值
            iou_thres: NMS IOU 閾值
            imgsz: 推論圖像尺寸
            max_det: 最大檢測數量 (設為 1 只顯示最高置信度物件)
            camera_id: 相機設備 ID
            camera_width: 相機捕獲寬度
            camera_height: 相機捕獲高度
            view_img: 是否顯示結果圖像
            use_gstreamer: 是否使用 GStreamer (Jetson 建議開啟)
            save_video: 是否儲存影片
            save_mask: 是否儲存遮罩圖
            output_dir: 輸出目錄
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.max_det = max_det
        self.view_img = view_img
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.save_video = save_video
        self.save_mask = save_mask
        self.output_dir = output_dir
        
        # 建立輸出目錄
        if save_video or save_mask:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 建立遮罩圖子目錄 (加上日期時間)
            if save_mask:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.mask_dir = self.output_dir / f'masks_{timestamp}'
                self.mask_dir.mkdir(parents=True, exist_ok=True)
                rospy.loginfo(f"遮罩圖儲存目錄: {self.mask_dir}")
        
        # 載入 YOLO11 TensorRT 模型
        rospy.loginfo(f"正在載入 YOLO11 TensorRT 引擎: {weights_path}")
        try:
            self.model = YOLO(weights_path, task="segment")
            
            # 檢查並設置類別名稱 (防呆機制)
            if not self.model.names:
                rospy.logwarn("模型未包含類別名稱，使用預設名稱")
                self.model.names = {0: 'hallway'}
            else:
                rospy.loginfo(f"模型類別: {self.model.names}")
                
        except Exception as e:
            rospy.logerr(f"載入模型時出錯：{e}")
            raise
        
        # 初始化相機
        rospy.loginfo(f"正在啟動相機 (ID: {camera_id})...")
        if use_gstreamer:
            pipeline = gstreamer_pipeline(
                sensor_id=camera_id,
                capture_width=camera_width,
                capture_height=camera_height,
                display_width=camera_width,
                display_height=camera_height
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        
        if not self.cap.isOpened():
            rospy.logerr("無法開啟相機，請檢查相機設備")
            raise RuntimeError("Camera initialization failed")
        
        # 驗證實際相機設定
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rospy.loginfo(f"相機成功啟動: {actual_w}x{actual_h}")
        
        # FPS 計算
        self.prev_time = 0
        self.fps = 0
        
        # 影片寫入器
        self.video_writer = None
        if save_video:
            self._init_video_writer(actual_w, actual_h)
        
        # 幀計數器 (用於遮罩檔名)
        self.frame_count = 0
    
    def _init_video_writer(self, width, height, fps=30):
        """初始化影片寫入器"""
        # 使用時間戳作為檔名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.output_dir / f"detection_{timestamp}.avi"
        
        # 使用 XVID 編碼器 (Ubuntu VLC 相容性最佳)
        # Jetson 原生支援，不需額外安裝編碼器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if self.video_writer.isOpened():
            rospy.loginfo(f"影片儲存路徑: {video_path}")
        else:
            rospy.logerr("無法初始化影片寫入器")
            self.save_video = False
    
    def predict(self, frame):
        """
        對單張圖像進行推論
        
        Args:
            frame: BGR 圖像 (numpy array)
            
        Returns:
            results: YOLO 推論結果
        """
        results = self.model.predict(
            frame, 
            conf=self.conf_thres, 
            iou=self.iou_thres, 
            verbose=False, 
            imgsz=self.imgsz,
            max_det=self.max_det  # 只保留最高置信度的物件
        )
        return results
    
    def process_masks(self, results, frame):
        """
        處理分割遮罩，找出道路參考點
        
        Args:
            results: YOLO 推論結果
            frame: 原始圖像
            
        Returns:
            mask_img: 二值化遮罩圖像
            road_detected: 是否檢測到道路
            forward_ref_x: 前方距離參考點 x 座標（遮罩最高點）
            forward_ref_y: 前方距離參考點 y 座標
            lateral_ref_x: 橫向誤差參考點 x 座標（畫面高度一半處）
            lateral_ref_y: 橫向誤差參考點 y 座標
        """
        height, width = frame.shape[:2]
        mask_img = np.zeros((height, width), dtype=np.uint8)
        road_detected = False
        forward_ref_x = width // 2
        forward_ref_y = height
        lateral_ref_x = width // 2
        lateral_ref_y = height // 2
        
        # 檢查是否有分割結果
        if results[0].masks is not None and len(results[0].masks) > 0:
            # 取得所有遮罩
            masks = results[0].masks.data.cpu().numpy()
            
            # 除錯：顯示遮罩原始尺寸（只顯示一次）
            if not hasattr(self, '_mask_size_logged'):
                rospy.loginfo(f"[DEBUG] 原始影像尺寸: {width} x {height}")
                rospy.loginfo(f"[DEBUG] YOLO 遮罩原始尺寸: {masks[0].shape}")
                self._mask_size_logged = True
            
            # 合併所有遮罩 (如果有多個)
            for mask in masks:
                mask_h, mask_w = mask.shape
                
                # 處理 letterbox 填充的情況
                # YOLO 將 640x480 影像填充到 640x640，上下各加 80 像素
                if mask_h == mask_w and mask_h != height:
                    # 計算 letterbox 的 padding
                    # 縮放因子 = min(mask_size/width, mask_size/height)
                    scale = min(mask_w / width, mask_h / height)
                    new_w = int(width * scale)
                    new_h = int(height * scale)
                    
                    # 上下 padding 量
                    pad_h = (mask_h - new_h) // 2
                    pad_w = (mask_w - new_w) // 2
                    
                    # 裁剪掉 padding 區域
                    mask_cropped = mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
                    
                    # 將裁剪後的遮罩調整到原始圖像尺寸
                    mask_resized = cv2.resize(mask_cropped, (width, height), interpolation=cv2.INTER_NEAREST)
                else:
                    # 遮罩尺寸與影像一致，直接 resize
                    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                
                mask_img[mask_resized > 0.5] = 255
            
            # 找出道路中心點（從底部往上搜尋）
            center_points = []
            for y in range(height - 1, -1, -1):
                row = mask_img[y, :]
                white_pixels = np.where(row == 255)[0]
                if len(white_pixels) > 0:
                    # 找出最大的連續白色區域
                    white_regions = np.split(white_pixels, np.where(np.diff(white_pixels) != 1)[0] + 1)
                    max_region = max(white_regions, key=lambda region: len(region))
                    center_x = int(np.mean(max_region))
                    center_points.append((center_x, y, len(max_region)))  # 加入寬度資訊
            
            if center_points:
                road_detected = True
                
                # 前方距離參考點：畫面水平中心與遮罩的最高交點（用於 e_d）
                # x 座標固定在畫面中心，y 座標為該垂直線上遮罩的最高點
                center_x_col = width // 2
                forward_ref_x = center_x_col  # x 固定在畫面中心
                forward_ref_y = height - 1  # 預設為畫面底部
                
                # 沿畫面中心垂直線搜尋遮罩最高點
                center_column = mask_img[:, center_x_col]
                white_pixels_in_col = np.where(center_column == 255)[0]
                if len(white_pixels_in_col) > 0:
                    # 最高點 = y 座標最小的位置
                    forward_ref_y = np.min(white_pixels_in_col)
                
                # 橫向誤差參考點：固定在停止距離對應的像素高度（用於 e_l）
                # y 座標固定不變，只有 x 座標根據該高度的遮罩中心變動
                target_height = getattr(self, 'lateral_ref_pixel_y', height // 2)
                lateral_ref_y = target_height  # y 座標固定
                lateral_ref_x = width // 2  # 預設為畫面中心
                
                # 找該高度行的遮罩中心點
                row = mask_img[target_height, :]
                white_pixels = np.where(row == 255)[0]
                if len(white_pixels) > 0:
                    # 找出最大的連續白色區域
                    white_regions = np.split(white_pixels, np.where(np.diff(white_pixels) != 1)[0] + 1)
                    max_region = max(white_regions, key=lambda region: len(region))
                    lateral_ref_x = int(np.mean(max_region))
        
        # 儲存遮罩圖
        if self.save_mask and road_detected:
            self._save_mask(mask_img)
        
        return mask_img, road_detected, forward_ref_x, forward_ref_y, lateral_ref_x, lateral_ref_y
    
    def _save_mask(self, mask_img):
        """儲存遮罩圖"""
        self.frame_count += 1
        mask_filename = self.mask_dir / f"mask_{self.frame_count:06d}.png"
        cv2.imwrite(str(mask_filename), mask_img)
    
    def draw_visualization(self, frame, results, mask_img, road_detected, 
                           forward_ref_x, forward_ref_y, lateral_ref_x=None, lateral_ref_y=None):
        """
        繪製視覺化結果
        
        Args:
            frame: 原始圖像
            results: YOLO 推論結果
            mask_img: 二值化遮罩
            road_detected: 是否檢測到道路
            forward_ref_x: 前方參考點 x 座標（藍色）
            forward_ref_y: 前方參考點 y 座標
            lateral_ref_x: 橫向參考點 x 座標（黃色）
            lateral_ref_y: 橫向參考點 y 座標
            
        Returns:
            annotated_frame: 標註後的圖像
        """
        # 使用 YOLO 內建的繪圖功能
        annotated_frame = results[0].plot()
        
        # 繪製參考點
        if road_detected:
            # 繪製藍色前方參考點
            cv2.circle(annotated_frame, (int(forward_ref_x), int(forward_ref_y)), 8, (255, 0, 0), -1)
            
            # 繪製黃色橫向參考點
            if lateral_ref_x is not None and lateral_ref_y is not None:
                cv2.circle(annotated_frame, (int(lateral_ref_x), int(lateral_ref_y)), 8, (0, 255, 255), -1)
            
            # 在圖像底部繪製黑色中心點
            image_height, image_width = annotated_frame.shape[:2]
            bottom_center_point = (image_width // 2, image_height - 1)
            cv2.circle(annotated_frame, bottom_center_point, 10, (0, 0, 0), -1)
        
        # 計算並顯示 FPS
        curr_time = time.time()
        if self.prev_time != 0:
            delta = curr_time - self.prev_time
            if delta > 0:
                self.fps = 1 / delta
        self.prev_time = curr_time
        
        cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame
    
    def write_video_frame(self, frame):
        """寫入影片幀"""
        if self.save_video and self.video_writer is not None:
            self.video_writer.write(frame)
    
    def read_frame(self):
        """讀取一幀圖像"""
        return self.cap.read()
    
    def release(self):
        """釋放資源"""
        if self.cap is not None:
            self.cap.release()
        
        # 釋放影片寫入器
        if self.video_writer is not None:
            self.video_writer.release()
            rospy.loginfo("影片儲存完成")
        
        # 輸出遮罩圖統計
        if self.save_mask:
            rospy.loginfo(f"共儲存 {self.frame_count} 張遮罩圖")
        
        cv2.destroyAllWindows()


def _get_default_paths():
    """取得預設路徑"""
    rospack = rospkg.RosPack()
    try:
        pkg_path = rospack.get_path('yolov7_ros')
    except rospkg.ResourceNotFound:
        pkg_path = os.path.dirname(os.path.abspath(__file__))
    
    default_weights = os.path.join(pkg_path, 'src/ultralytics/runs/segment/yolo11_seg_model12/weights/best.engine')
    default_output = os.path.join(pkg_path, 'outputs')
    return default_weights, default_output


def run(
    weights=None,
    source=0,
    imgsz=640,
    conf_thres=0.5,
    iou_thres=0.5,
    max_det=1,
    device='0',
    view_img=True,
    camera_width=640,
    camera_height=480,
    use_gstreamer=True,
    save_video=False,
    save_mask=False,
    output_dir=None,
):
    """
    主運行函數
    
    Args:
        weights: TensorRT 引擎檔案路徑
        source: 相機設備 ID
        imgsz: 推論圖像尺寸
        conf_thres: 置信度閾值
        iou_thres: NMS IOU 閾值
        max_det: 最大檢測數量 (1 = 只顯示最高置信度物件)
        device: CUDA 設備
        view_img: 是否顯示結果
        camera_width: 相機寬度
        camera_height: 相機高度
        use_gstreamer: 是否使用 GStreamer
        save_video: 是否儲存影片
        save_mask: 是否儲存遮罩圖
        output_dir: 輸出目錄
    """
    # 創建優雅退出處理器
    killer = GracefulKiller()
    
    # 取得預設路徑
    default_weights, default_output = _get_default_paths()
    if weights is None:
        weights = default_weights
    if output_dir is None:
        output_dir = default_output
    
    # 初始化 ROS 節點和發布器
    if not rospy.core.is_initialized():
        rospy.init_node('road_detection_node', anonymous=True)
    
    road_info_pub = rospy.Publisher('/road_info', Float32MultiArray, queue_size=10)
    avoidance_pub = rospy.Publisher('/avoidance_enabled', String, queue_size=10)
    rate = rospy.Rate(30)  # 30Hz
    
    # 避障控制狀態
    avoidance_enabled = False
    
    # 初始化檢測器
    try:
        detector = YOLO11RoadDetector(
            weights_path=weights,
            device=device,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            imgsz=imgsz,
            max_det=max_det,
            camera_id=int(source) if str(source).isnumeric() else 0,
            camera_width=camera_width,
            camera_height=camera_height,
            view_img=view_img,
            use_gstreamer=use_gstreamer,
            save_video=save_video,
            save_mask=save_mask,
            output_dir=output_dir
        )
    except Exception as e:
        rospy.logerr(f"初始化檢測器失敗: {e}")
        return
    
    # 初始化 BEV 轉換器
    try:
        bev_transformer = BEVTransformer()
        rospy.loginfo("BEV 轉換器初始化成功")
        
        # 計算停止距離對應的像素 y 座標（用於橫向參考點）
        lateral_ref_distance = 1.1  # 公尺（停止距離）
        lateral_ref_pixel_y = bev_transformer.distance_to_pixel_y(lateral_ref_distance)
        detector.lateral_ref_pixel_y = lateral_ref_pixel_y
        rospy.loginfo(f"橫向參考點距離: {lateral_ref_distance}m -> 像素 y: {lateral_ref_pixel_y}")
        
    except Exception as e:
        rospy.logerr(f"初始化 BEV 轉換器失敗: {e}")
        return
    
    rospy.loginfo("YOLO11 道路檢測節點啟動")
    rospy.loginfo("按 's' 鍵開始/暫停避障，按 'q' 鍵離開")
    
    # 創建視窗
    window_name = "YOLO11n-seg Road Detection"
    bev_window_name = "BEV View"
    if view_img:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(bev_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(bev_window_name, 400, 480)
    
    # FPS 統計
    total_frame_count = 0
    start_time = time.time()
    recent_frame_count = 0
    log_interval = 5.0  # 每 5 秒記錄一次
    last_log_time = time.time()
    
    # 建立 FPS log 檔案
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fps_log_path = log_dir / f"fps_log_{log_timestamp}.txt"
    
    # 寫入 log 標頭
    with open(fps_log_path, 'w', encoding='utf-8') as f:
        f.write(f"# FPS Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# Format: timestamp, current_fps, average_fps\n")
        f.write("-" * 50 + "\n")
    
    rospy.loginfo(f"FPS log 檔案: {fps_log_path}")
    
    try:
        while not killer.kill_now and not rospy.is_shutdown():
            # 讀取圖像
            success, frame = detector.read_frame()
            
            if not success:
                rospy.logwarn("無法讀取相機畫面")
                continue
            
            # 進行推論
            results = detector.predict(frame)
            
            # 處理遮罩並找出參考點（雙參考點）
            mask_img, road_detected, forward_ref_x, forward_ref_y, lateral_ref_x, lateral_ref_y = detector.process_masks(results, frame)
            
            # FPS 計算
            total_frame_count += 1
            recent_frame_count += 1
            current_time = time.time()
            
            # 定期記錄 FPS
            if current_time - last_log_time >= log_interval:
                # 即時 FPS（最近 5 秒）
                recent_fps = recent_frame_count / log_interval
                # 平均 FPS（從啟動開始）
                avg_fps = total_frame_count / (current_time - start_time)
                
                rospy.loginfo(f"[FPS] Current: {recent_fps:.1f} | Average: {avg_fps:.1f}")
                
                # 寫入 log 檔案
                with open(fps_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().strftime('%H:%M:%S')}, {recent_fps:.1f}, {avg_fps:.1f}\n")
                
                # 重置即時計數
                last_log_time = current_time
                recent_frame_count = 0
            
            # 準備 ROS 消息
            road_info_msg = Float32MultiArray()
            height, width = frame.shape[:2]
            
            if road_detected:
                # 使用 BEV 轉換計算誤差值（分離參考點）
                errors = bev_transformer.compute_errors(
                    forward_ref_x, forward_ref_y,  # 前方距離參考點
                    lateral_ref_x, lateral_ref_y   # 橫向誤差參考點
                )
                
                # 新格式: [road_detected, e_d, e_d_dot, e_l, e_l_dot, y_ground, x_ground]
                road_info_msg.data = [
                    1,                    # road_detected
                    errors['e_d'],        # 前方距離誤差
                    errors['e_d_dot'],    # 前方距離變化率
                    errors['e_l'],        # 橫向誤差
                    errors['e_l_dot'],    # 橫向變化率
                    errors['y_ground'],   # 前方距離 (公尺)
                    errors['x_ground']    # 橫向距離 (公尺)
                ]
            else:
                # 未檢測到道路，重置 BEV 狀態
                bev_transformer.reset_state()
                road_info_msg.data = [0, 0, 0, 0, 0, 0, 0]
            
            # 發布道路資訊
            road_info_pub.publish(road_info_msg)
            
            # 視覺化（顯示前方參考點藍色、橫向參考點黃色）
            annotated_frame = detector.draw_visualization(
                frame, results, mask_img, road_detected, 
                forward_ref_x, forward_ref_y, lateral_ref_x, lateral_ref_y
            )
            
            # 儲存影片幀
            if save_video:
                detector.write_video_frame(annotated_frame)
            
            # 顯示
            if view_img:
                cv2.imshow(window_name, annotated_frame)
                
                # 建立並顯示 BEV 鳥瞰圖
                if road_detected:
                    bev_view = bev_transformer.create_bev_visualization(
                        frame,
                        forward_ref_x, forward_ref_y,
                        lateral_ref_x, lateral_ref_y
                    )
                else:
                    # 未檢測到道路時仍顯示 BEV 視窗（無參考點）
                    bev_view = bev_transformer.create_bev_visualization(frame)
                cv2.imshow(bev_window_name, bev_view)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    rospy.loginfo("使用者按下 'q' 鍵，準備退出...")
                    break
                elif key == ord('s'):
                    avoidance_enabled = not avoidance_enabled
                    status = "啟動" if avoidance_enabled else "暫停"
                    rospy.loginfo(f"避障功能已{status}")
                    # 發布避障狀態
                    avoidance_pub.publish("enabled" if avoidance_enabled else "disabled")
            
            rate.sleep()
            
    except Exception as e:
        rospy.logerr(f"推論過程中出現錯誤：{e}")
    finally:
        # 寫入最終摘要
        total_time = time.time() - start_time
        final_avg_fps = total_frame_count / total_time if total_time > 0 else 0
        
        with open(fps_log_path, 'a', encoding='utf-8') as f:
            f.write("-" * 50 + "\n")
            f.write(f"# Summary\n")
            f.write(f"Total Frames: {total_frame_count}\n")
            f.write(f"Total Time: {total_time:.1f}s\n")
            f.write(f"Average FPS: {final_avg_fps:.1f}\n")
        
        rospy.loginfo(f"[Summary] Total: {total_frame_count} frames, Avg FPS: {final_avg_fps:.1f}")
        
        # 釋放資源
        detector.release()
        rospy.loginfo("道路檢測節點已完成運行")


def parse_opt():
    """解析命令列參數"""
    # 取得預設路徑
    default_weights, default_output = _get_default_paths()
    
    parser = argparse.ArgumentParser(description='YOLO11n-seg TensorRT 道路檢測節點')
    
    # 模型參數
    parser.add_argument('--weights', type=str, default=default_weights,
                       help='TensorRT 引擎檔案路徑')
    parser.add_argument('--source', type=str, default='0', 
                       help='相機設備 ID')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640,
                       help='推論圖像尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.5,
                       help='置信度閾值')
    parser.add_argument('--iou-thres', type=float, default=0.5,
                       help='NMS IOU 閾值')
    parser.add_argument('--max-det', type=int, default=1,
                       help='最大檢測數量')
    parser.add_argument('--device', default='0', help='CUDA 設備')
    
    # 顯示參數
    parser.add_argument('--view-img', type=str, default='True',
                       help='是否顯示結果圖像')
    
    # 相機參數
    parser.add_argument('--camera-width', type=int, default=640)
    parser.add_argument('--camera-height', type=int, default=480)
    parser.add_argument('--use-gstreamer', type=str, default='True')
    
    # 儲存參數
    parser.add_argument('--save-video', type=str, default='True')
    parser.add_argument('--save-mask', type=str, default='True')
    parser.add_argument('--output-dir', type=str, default=default_output,
                       help='輸出目錄')
    
    # 忽略未知的 ROS 參數
    args, unknown = parser.parse_known_args()
    
    # 從 ROS 參數服務器獲取覆蓋值
    if rospy.has_param('~weights'):
        args.weights = rospy.get_param('~weights')
    if rospy.has_param('~source'):
        args.source = rospy.get_param('~source')
    if rospy.has_param('~view_img'):
        args.view_img = rospy.get_param('~view_img')
    if rospy.has_param('~conf_thres'):
        args.conf_thres = rospy.get_param('~conf_thres')
    if rospy.has_param('~iou_thres'):
        args.iou_thres = rospy.get_param('~iou_thres')
    if rospy.has_param('~max_det'):
        args.max_det = rospy.get_param('~max_det')
    if rospy.has_param('~camera_width'):
        args.camera_width = rospy.get_param('~camera_width')
    if rospy.has_param('~camera_height'):
        args.camera_height = rospy.get_param('~camera_height')
    if rospy.has_param('~use_gstreamer'):
        args.use_gstreamer = rospy.get_param('~use_gstreamer')
    if rospy.has_param('~imgsz'):
        args.imgsz = rospy.get_param('~imgsz')
    if rospy.has_param('~save_video'):
        args.save_video = rospy.get_param('~save_video')
    if rospy.has_param('~save_mask'):
        args.save_mask = rospy.get_param('~save_mask')
    if rospy.has_param('~output_dir'):
        args.output_dir = rospy.get_param('~output_dir')
    
    # 處理布林值字串
    if isinstance(args.view_img, str):
        args.view_img = args.view_img.lower() == 'true'
    if isinstance(args.use_gstreamer, str):
        args.use_gstreamer = args.use_gstreamer.lower() == 'true'
    if isinstance(args.save_video, str):
        args.save_video = args.save_video.lower() == 'true'
    if isinstance(args.save_mask, str):
        args.save_mask = args.save_mask.lower() == 'true'
    
    return args


def main():
    """主函數"""
    opt = parse_opt()
    
    rospy.loginfo("=" * 50)
    rospy.loginfo("YOLO11n-seg TensorRT 道路檢測節點")
    rospy.loginfo("=" * 50)
    rospy.loginfo(f"權重檔案: {opt.weights}")
    rospy.loginfo(f"相機來源: {opt.source}")
    rospy.loginfo(f"推論尺寸: {opt.imgsz}")
    rospy.loginfo(f"置信度閾值: {opt.conf_thres}")
    rospy.loginfo(f"IOU 閾值: {opt.iou_thres}")
    rospy.loginfo(f"最大檢測數量: {opt.max_det}")
    rospy.loginfo(f"相機解析度: {opt.camera_width}x{opt.camera_height}")
    rospy.loginfo(f"使用 GStreamer: {opt.use_gstreamer}")
    rospy.loginfo(f"顯示圖像: {opt.view_img}")
    rospy.loginfo(f"儲存影片: {opt.save_video}")
    rospy.loginfo(f"儲存遮罩: {opt.save_mask}")
    if opt.save_video or opt.save_mask:
        rospy.loginfo(f"輸出目錄: {opt.output_dir}")
    rospy.loginfo("=" * 50)
    
    run(
        weights=opt.weights,
        source=opt.source,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        max_det=opt.max_det,
        device=opt.device,
        view_img=opt.view_img,
        camera_width=opt.camera_width,
        camera_height=opt.camera_height,
        use_gstreamer=opt.use_gstreamer,
        save_video=opt.save_video,
        save_mask=opt.save_mask,
        output_dir=opt.output_dir,
    )


if __name__ == "__main__":
    main()