# Vision Fuzzy Control — ROS Catkin Workspace

> **影像式模糊控制避障系統（結合語音控制）**  
> 基於 YOLO11 TensorRT + Bird's-Eye View + 四輸入 FLC + LLM 語音指令的醫療照護機器人自主導航系統。  
> 平台：**NVIDIA Jetson Orin NX** ・ ROS：**Noetic** ・ 論文研究專案（NSTC）

---

## 工作空間結構

```
ros/                                   ← Catkin workspace 根目錄
├── src/
│   ├── yolov7_ros/                    ← 主套件（核心系統）★
│   │   ├── scripts/                   ← 所有 ROS 節點與輔助腳本
│   │   ├── config/                    ← 相機標定、語音 LLM Prompt 設定
│   │   ├── launch/
│   │   │   └── integrated_system.launch
│   │   └── src/                       ← 第三方程式庫（ultralytics、yolov7）
│   ├── detection_msgs/                ← 自定義 ROS 訊息型別
│   └── rf2o_laser_odometry/           ← 雷射里程計（軌跡 Ground Truth 比對用）
├── build/                             ← Catkin 編譯目錄（gitignore）
├── devel/                             ← Catkin 開發目錄（gitignore）
└── README.md                          ← 本文件
```

---

## 套件說明

| 套件 | 說明 |
|------|------|
| **`yolov7_ros`** | 主套件。YOLO11 TRT 道路分割、BEV 透視轉換、四輸入模糊控制器、語音指令節點、整合控制器、資料記錄、資源監控。詳見 [`src/yolov7_ros/README.md`](src/yolov7_ros/README.md) |
| `detection_msgs` | 自定義 ROS 訊息型別，供 YOLO 偵測結果使用 |
| `rf2o_laser_odometry` | RF2O 雷射掃描里程計（稠密掃描對齊，0.9 ms/frame），用於實驗軌跡 Ground Truth 比對。[論文：ICRA 2016](http://mapir.isa.uma.es/work/rf2o) |

---

## 快速開始

### 1. 建置工作空間

```bash
cd ~/ros
catkin_make
source devel/setup.bash
```

### 2. 啟動整合系統

```bash
roslaunch yolov7_ros integrated_system.launch
```

### 3. 常用啟動選項

```bash
# 關閉可選節點（最小系統）
roslaunch yolov7_ros integrated_system.launch \
    enable_logger:=false \
    enable_resource_monitor:=false
```

---

## 系統架構概覽

```
Camera → [road_detection_node] → /road_info
                                      ↓
                         [fuzzy_controller_node] → /fuzzy_cmd_vel
                                                          ↓
Microphone → [voice_command_node] → /voice_command_code   ↓
                                                          ↓
                              [integrated_controller_node] → /cmd_vel → 底盤
```

詳細說明請參考 [`src/yolov7_ros/README.md`](src/yolov7_ros/README.md)。

---

## 主要 ROS Topics

| Topic | 型別 | 說明 |
|-------|------|------|
| `/road_info` | `Float32MultiArray` | `[road_det, e_d, e_d_dot, e_l, e_l_dot, y_m, x_m]` |
| `/fuzzy_cmd_vel` | `Twist` | 模糊控制器輸出速度 |
| `/voice_command_code` | `String` | 5 位數語音指令碼 |
| `/avoidance_enabled` | `String` | `"enabled"` / `"disabled"` |
| `/system_status` | `String` | 系統執行狀態 |
| `/cmd_vel` | `Twist` | 最終底盤速度命令 |

---

## 系統需求

| 項目 | 規格 |
|------|------|
| 硬體 | NVIDIA Jetson Orin NX（JetPack 5.x）|
| OS | Ubuntu 20.04 |
| ROS | Noetic |
| Python | 3.8+ |
| 主要 Python 套件 | `ultralytics`, `opencv-python`, `pyaudio`, `pydub`, `soundfile`, `noisereduce`, `requests` |

---

## .gitignore 說明

| 忽略路徑 | 原因 |
|---------|------|
| `build/`, `devel/` | Catkin 編譯產物 |
| `src/yolov7_ros/outputs/` | 執行期輸出（CSV、影片、log）|
| `src/yolov7_ros/recordings/` | 語音錄音暫存 |
| `src/yolov7_ros/src/` | 第三方程式庫（ultralytics、yolov7）|
| `src/yolov7_ros/docs/`, `include/` | 暫不追蹤的文件與標頭 |
| `record.txt`, `frames.pdf` | 實驗紀錄雜項 |

---

## 相關文件

- 主套件完整說明：[`src/yolov7_ros/README.md`](src/yolov7_ros/README.md)  
- 語音接收模組說明：[`src/yolov7_ros/scripts/VOICE_RECEIVER_README.md`](src/yolov7_ros/scripts/VOICE_RECEIVER_README.md)  
- 模糊控制規則：`src/yolov7_ros/src/fuzzy_control_design/fuzzy_rules_relaxed.csv`  
- 相機標定設定：`src/yolov7_ros/config/camera_config.yaml`  
- LLM 語音 Prompt：`src/yolov7_ros/config/voice_llm_prompt_config.json`
