#!/usr/bin/python3

import argparse
import os
import platform
import sys
import signal
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import rospy
from std_msgs.msg import Float32MultiArray

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 添加優雅退出的支持
class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        self.kill_now = True
        rospy.loginfo("接收到終止信號，準備退出...")

# 修改：添加過濾ROS參數的功能
def filter_ros_args():
    """過濾掉ROS相關的參數，只保留腳本自身的參數"""
    filtered_args = []
    i = 0
    while i < len(sys.argv):
        # 跳過ROS參數（通常以雙下劃線開頭）
        if sys.argv[i].startswith('__'):
            i += 1
            # 如果參數有值，再跳過一個
            if i < len(sys.argv) and not sys.argv[i].startswith('-'):
                i += 1
            continue
        filtered_args.append(sys.argv[i])
        i += 1
    return filtered_args

# 在解析參數之前過濾掉ROS參數
sys.argv = filter_ros_args()

# 添加資源清理函數
def cleanup_resources(video_writers):
    """釋放資源"""
    rospy.loginfo("釋放資源...")
    
    # 關閉所有視頻寫入器
    for writer in video_writers:
        if isinstance(writer, cv2.VideoWriter):
            writer.release()
    
    # 關閉所有OpenCV窗口
    cv2.destroyAllWindows()
    
    rospy.loginfo("資源釋放完成")

# 從環境變數或參數設置ROOT路徑
def get_root_path():
    # 嘗試從環境變數獲取路徑
    env_root = os.environ.get('YOLOV7_ROOT')
    if env_root:
        return Path(env_root)
    
    # 使用默認路徑
    default_root = Path('/home/cir/ros/src/yolov7_ros/src/yolov7/yolov7-u7/seg/')
    return default_root

ROOT = get_root_path()  # YOLOv7 根目錄
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 將 ROOT 加入 PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相對路徑

# 導入YOLO模型相關模塊（如果路徑正確）
try:
    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
    from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                            increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
    from utils.plots import Annotator, colors, save_one_box
    from utils.segment.general import process_mask, scale_masks
    from utils.segment.plots import plot_masks
    from utils.torch_utils import select_device, smart_inference_mode
except ImportError as e:
    rospy.logerr(f"導入YOLO模塊時出錯：{e}")
    rospy.logerr(f"當前ROOT路徑：{ROOT}")
    rospy.logerr("請檢查YOLOV7_ROOT環境變數或路徑設置")
    sys.exit(1)

@smart_inference_mode()
def run(
        weights=ROOT / 'runs/train-seg/exp32/weights/best.pt',  # 模型路徑
        source='0',  # 文件/目錄/URL/通配符，0表示攝像頭
        data=ROOT / 'data-seg4/data.yaml',  # dataset.yaml 路徑
        imgsz=(640, 640),  # 推理尺寸（高，寬）
        conf_thres=0.2,  # 置信度閾值
        iou_thres=0.2,  # NMS IOU 閾值
        max_det=1,  # 每張圖像的最大檢測數量
        device='0',  # CUDA 設備，例如 0 或 0,1,2,3 或 cpu
        view_img=False,  # 顯示結果
        save_txt=False,  # 將結果保存為 *.txt
        save_conf=False,  # 在 --save-txt 標籤中保存置信度
        save_crop=False,  # 保存裁剪的預測框
        nosave=False,  # 不保存圖像/視頻
        classes=None,  # 按類別過濾：--class 0，或 --class 0 2 3
        agnostic_nms=False,  # 類別無關的 NMS
        augment=False,  # 增強推理
        visualize=False,  # 可視化特徵
        update=False,  # 更新所有模型
        project=ROOT / 'runs/predict-seg',  # 將結果保存到 project/name
        name='exp',  # 將結果保存到 project/name
        exist_ok=False,  # 已存在的 project/name 可以，不自動增加
        line_thickness=2,  # 邊框厚度（像素）
        hide_labels=False,  # 隱藏標籤
        hide_conf=False,  # 隱藏置信度
        half=False,  # 使用 FP16 半精度推理
        dnn=False,  # 對於 ONNX 推理，使用 OpenCV DNN
        camera_width=640,  # 相機捕獲寬度
        camera_height=480,  # 相機捕獲高度
):
    # 創建優雅退出處理器
    killer = GracefulKiller()
    
    # 初始化 ROS 節點和發布器
    # 如果ROS節點已經初始化，則不再初始化
    if not rospy.core.is_initialized():
        rospy.init_node('road_detection_node', anonymous=True)
    road_info_pub = rospy.Publisher('/road_info', Float32MultiArray, queue_size=10)
    rate = rospy.Rate(10)  # 10Hz

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # 保存推理圖像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # 下載

    # 目錄
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增量運行
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 創建目錄
    mask_dir = save_dir / 'masks'
    mask_dir.mkdir(parents=True, exist_ok=True)  # 創建遮罩目錄

    # 加載模型
    try:
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # 檢查圖像尺寸
    except Exception as e:
        rospy.logerr(f"加載模型時出錯：{e}")
        return

    # # 數據加載器
    # try:
    #     if webcam:
    #         view_img = check_imshow()
    #         cudnn.benchmark = True  # 設置為 True 以加速固定圖像尺寸的推理
    #         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    #         bs = len(dataset)  # 批量大小
    #     else:
    #         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    #         bs = 1  # 批量大小
    #     vid_path, vid_writer = [None] * bs, [None] * bs
    # except Exception as e:
    #     rospy.logerr(f"初始化數據加載器時出錯：{e}")
    #     return
        # 數據加載器
    try:
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # 設置為 True 以加速固定圖像尺寸的推理
            
            # 使用帶有相機分辨率參數的LoadStreams
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, 
                                  camera_width=camera_width, camera_height=camera_height)
            bs = len(dataset)  # 批量大小
            
            # 記錄相機設置信息
            rospy.loginfo(f"相機設置為 {camera_width}x{camera_height}")
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # 批量大小
        vid_path, vid_writer = [None] * bs, [None] * bs
    except Exception as e:
        rospy.logerr(f"初始化數據加載器時出錯：{e}")
        return

    # 初始化 FPS 計算
    startTime = time.time()

    # 運行推理
    try:
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # 預熱
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        
        for path, im, im0s, vid_cap, s in dataset:
            # 檢查是否收到退出信號
            if killer.kill_now or rospy.is_shutdown():
                rospy.loginfo("檢測到退出信號，正在停止處理...")
                break
                
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # 擴展批量維度

            # 推理
            with dt[1]:
                visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred, out = model(im, augment=augment, visualize=visualize_path)
                proto = out[1]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

            # 處理預測結果
            for i, det in enumerate(pred):  # 每張圖像
                seen += 1
                if webcam:  # 批量大小 >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # 轉換為 Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # 打印字符串
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 正規化增益 whwh
                imc = im0.copy() if save_crop else im0  # 用於 save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                mask_img = None  # 初始化 mask_img
                road_detected = False  # 初始化路面檢測狀態

                # 創建用於發布的 Float32MultiArray 消息
                road_info_msg = Float32MultiArray()
                road_info_msg.data = [0, 0, 0]  # [是否檢測到道路(0/1), 目標x座標, 圖像寬度]

                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                    # 將框從 img_size 縮放到 im0 尺寸
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # 遮罩生成和保存
                    mask_img = np.zeros((im0.shape[0], im0.shape[1]), dtype=np.uint8)
                    for mask in masks:
                        mask_np = mask.cpu().numpy()  # 確保遮罩是 numpy 數組
                        mask_resized = cv2.resize(mask_np, (im0.shape[1], im0.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                        mask_img[mask_resized > 0.5] = 255

                    if dataset.mode == 'video':
                        frame_number = vid_cap.get(cv2.CAP_PROP_POS_FRAMES)
                        mask_save_path = str(mask_dir / f"{p.stem}_frame{int(frame_number)}_mask.png")
                    else:
                        mask_save_path = str(mask_dir / f"{p.stem}_mask.png")

                    cv2.imwrite(mask_save_path, mask_img)

                    # 打印結果
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # 每個類別的檢測數量
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串

                    # 遮罩繪製
                    mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                    im_masks = plot_masks(im[i], masks, mcolors)  # 帶有遮罩的圖像 shape(imh,imw,3)
                    annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # 縮放到原始 h, w

                    # 寫入結果
                    for *xyxy, conf, cls in reversed(det[:, :6]):
                        if save_txt:  # 寫入文件
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # 正規化 xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 標籤格式
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # 在圖像上添加邊框
                            c = int(cls)  # 整數類別
                            label = None if hide_labels else (
                                names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc,
                                        file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # 流式結果
                im0 = annotator.result()

                # 對於每一幀，找到可駕駛道路的中心點並繪製相關點
                if mask_img is not None:
                    height, width = mask_img.shape
                    center_points = []

                    # 從下往上掃描，找到中心點
                    for y in range(height - 1, -1, -1):
                        row = mask_img[y, :]
                        white_pixels = np.where(row == 255)[0]
                        if len(white_pixels) > 0:
                            white_regions = np.split(white_pixels, np.where(np.diff(white_pixels) != 1)[0] + 1)
                            max_region = max(white_regions, key=lambda region: len(region))
                            center_x = int(np.mean(max_region))
                            center_points.append((center_x, y))

                    if center_points:
                        road_detected = True
                        
                        # 計算紅色遮罩圖最高點
                        highest_point_y = min(point[1] for point in center_points)
                        
                        # 選擇參考點
                        if highest_point_y >= int(height * 0.35):
                            # 在圖像高度的 30% 位置處獲取紅色遮罩圖的中心點
                            desired_y = int(height * 0.3)
                            # 找到在 desired_y 處的紅色遮罩圖的中心點
                            row = mask_img[desired_y, :]
                            white_pixels = np.where(row == 255)[0]
                            if len(white_pixels) > 0:
                                white_regions = np.split(white_pixels, np.where(np.diff(white_pixels) != 1)[0] + 1)
                                max_region = max(white_regions, key=lambda region: len(region))
                                desired_x = int(np.mean(max_region))
                                cv2.circle(im0, (int(desired_x), desired_y), 8, (255, 0, 0), -1)  # 繪製紅色圓點在 desired_y
                            else:
                                # 如果在 desired_y 處沒有找到紅色遮罩，使用最低點
                                closest_point = min(center_points, key=lambda point: abs(point[1] - desired_y))
                                desired_x = closest_point[0]
                                desired_y = closest_point[1]
                                cv2.circle(im0, (int(desired_x), desired_y), 8, (255, 0, 0), -1)
                        else:
                            # 按照原先的方法處理
                            target_y = int(height / 2)  # 圖像高度的一半處
                            closest_point = min(center_points, key=lambda point: abs(point[1] - target_y))
                            desired_x = closest_point[0]
                            desired_y = closest_point[1]
                            cv2.circle(im0, (int(desired_x), desired_y), 8, (255, 0, 0), -1)  # 繪製紅色圓點
                        
                        # 更新要發布的數據
                        road_info_msg.data = [1, desired_x, width]
                
                # 在圖像底部繪製一個黑色的中心點
                if road_detected:
                    image_height, image_width = im0.shape[:2]
                    bottom_center_point = (image_width // 2, image_height - 1)  # 底部中心點
                    cv2.circle(im0, bottom_center_point, 10, (0, 0, 0), -1)  # 繪製黑色圓點
                    
                # 發布道路資訊
                road_info_pub.publish(road_info_msg)

                # 在圖像上顯示 FPS
                currentTime = time.time()
                fps = 1 / (currentTime - startTime)
                startTime = currentTime

                cv2.putText(im0, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 255, 255), 2)

                if view_img:
                    try:
                        if platform.system() == 'Linux' and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 毫秒
                    except Exception as e:
                        rospy.logwarn(f"顯示圖像時出錯：{e}")

                # 保存結果（帶有檢測結果的圖像）
                if save_img:
                    try:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' 或 'stream'
                            if vid_path[i] != save_path:  # 新視頻
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # 釋放之前的視頻寫入器
                                if vid_cap:  # 視頻
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # 流
                                    fps, w, h = 10, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix('.avi'))
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'),
                                                            fps, (w, h))
                            vid_writer[i].write(im0)
                    except Exception as e:
                        rospy.logwarn(f"保存圖像時出錯：{e}")

            # 打印時間（僅推理）
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            
            # 循環率控制
            rate.sleep()
            
            # 再次檢查是否收到退出信號
            if killer.kill_now or rospy.is_shutdown():
                rospy.loginfo("檢測到退出信號，正在停止處理...")
                break
    
    except Exception as e:
        rospy.logerr(f"推理過程中出現錯誤：{e}")
    finally:
        # 釋放資源
        cleanup_resources(vid_writer)
        rospy.loginfo("道路檢測節點已完成運行")

    # 打印結果
    t = tuple(x.t / seen * 1E3 for x in dt)  # 每張圖像的速度
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # 更新模型（修复 SourceChangeWarning）

def parse_opt():
    """修改參數解析以適配ROS啟動"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'runs/train-seg/exp32/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data-seg4/data.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default='True', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        # 添加相機分辨率參數
    parser.add_argument('--camera-width', type=int, default=640, help='設置相機捕獲寬度')
    parser.add_argument('--camera-height', type=int, default=480, help='設置相機捕獲高度')
    
    # 忽略ROS參數
    args, unknown = parser.parse_known_args()
    
    # 檢查是否有ROS參數設置覆蓋
    if rospy.has_param('~view_img'):
        args.view_img = rospy.get_param('~view_img')
    if rospy.has_param('~conf_thres'):
        args.conf_thres = rospy.get_param('~conf_thres')
    if rospy.has_param('~iou_thres'):
        args.iou_thres = rospy.get_param('~iou_thres')
    if rospy.has_param('~source'):
        args.source = rospy.get_param('~source')
    if rospy.has_param('~camera_width'):
        args.camera_width = rospy.get_param('~camera_width')
    if rospy.has_param('~camera_height'):
        args.camera_height = rospy.get_param('~camera_height')
    
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    print_args(vars(args))
    return args

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
