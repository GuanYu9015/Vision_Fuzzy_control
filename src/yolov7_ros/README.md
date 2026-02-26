# 影像式模糊控制避障系統（結合語音控制）

> **Visual Fuzzy Obstacle Avoidance System with Voice Control**  
> 基於 YOLO11 TensorRT 實例分割 + 四輸入模糊邏輯控制器 + 語音指令整合的機器人自主導航系統。  
> 平台：**NVIDIA Jetson Orin NX** ・ 框架：**ROS 1 (Noetic)** ・ 啟動指令：`roslaunch yolov7_ros integrated_system.launch`

---

## 目錄

1. [系統概覽](#1-系統概覽)
2. [架構說明](#2-架構說明)
3. [ROS Nodes 與 Topics](#3-ros-nodes-與-topics)
4. [各模組詳細說明](#4-各模組詳細說明)
5. [模糊控制器設計](#5-模糊控制器設計)
6. [語音指令編碼](#6-語音指令編碼)
7. [安裝與依賴](#7-安裝與依賴)
8. [啟動方式](#8-啟動方式)
9. [輸出檔案說明](#9-輸出檔案說明)
10. [調參指引](#10-調參指引)

---

## 1. 系統概覽

本系統為論文研究專案，實現以下功能：

- **影像辨識**：透過 USB 相機擷取畫面，使用 YOLO11n-seg TensorRT 引擎進行即時道路/走廊實例分割
- **BEV 轉換**：將影像座標透過 Bird's-Eye View (BEV) 幾何轉換為實際距離（公尺）
- **模糊控制**：以 4 個誤差輸入（`e_d`, `e_d_dot`, `e_l`, `e_l_dot`）推論機器人速度命令
- **語音控制**：藉由藍牙麥克風錄音、VAD 偵測、雲端 LLM 語音辨識，將自然語言轉換為 5 位數指令碼
- **整合控制**：仲裁語音指令與模糊控制的優先權，輸出最終 `/cmd_vel` 命令

### 整體資料流

```
USB Camera
    │
    ▼
[road_detection_node]  ── YOLO11n-seg TRT + BEV ──►  /road_info
                                                          │
                                              ┌───────────┘
                                              ▼
                                    [fuzzy_controller_node]
                                    4-Input FLC ───────►  /fuzzy_cmd_vel
                                                               │
[voice_command_node]  ── LLM STT ──►  /voice_command_code     │
                    │                        │                 │
                    └────────────────────────┴─────────────────┤
                                                               ▼
                                             [integrated_controller_node]
                                             仲裁邏輯 ──────►  /cmd_vel ──► 機器人底盤
```

---

## 2. 架構說明

### 核心設計原則

| 原則 | 說明 |
|------|------|
| **感知-推論-控制分離** | 各節點單一職責，透過 ROS Topics 非同步通訊 |
| **語音優先、模糊次之** | 語音指令執行期間，`integrated_controller_node` 暫停模糊控制輸出 |
| **鍵盤/語音雙重觸發** | `/avoidance_enabled` topic 支援按鍵 `s` 與語音指令同時控制避障開關 |
| **低通濾波平滑輸出** | EMA 濾波（`α_v=0.2`, `α_ω=0.3`）抑制速度抖動 |

---

## 3. ROS Nodes 與 Topics

### Nodes

| Node 名稱 | 腳本 | 功能 |
|-----------|------|------|
| `road_detection_node` | `mod_predict_yolo11_trt.py` | YOLO 分割 + BEV 誤差計算 |
| `fuzzy_controller_node` | `mod_fuzzy_control4.py` | 四輸入模糊推論 |
| `voice_command_node` | `mod_voice_processing4.py` | 語音錄音、辨識、指令發布 |
| `integrated_controller_node` | `integrated_control4.py` | 指令仲裁與最終 cmd_vel 輸出 |
| `fuzzy_data_logger` | `fuzzy_data_logger.py` | 資料記錄（可選，`enable_logger:=true`）|
| `resource_monitor_node` | `resource_monitor_node.py` | 系統資源監控（可選，`enable_resource_monitor:=true`）|

### Topics

| Topic | 型別 | 方向 | 說明 |
|-------|------|------|------|
| `/road_info` | `Float32MultiArray` | `road_detection_node` → others | `[road_detected, e_d, e_d_dot, e_l, e_l_dot, y_ground, x_ground]` |
| `/fuzzy_cmd_vel` | `Twist` | `fuzzy_controller_node` → integrated | 模糊推論輸出速度 |
| `/voice_command_code` | `String` | `voice_command_node` → others | 5 位數指令碼（如 `11000`）|
| `/avoidance_enabled` | `String` | 雙向 | `"enabled"` / `"disabled"` 控制避障開關 |
| `/system_status` | `String` | 雙向 | `"voice_command_executing"` / `"fuzzy_control_active"` / `"avoidance_paused"` |
| `/cmd_vel` | `Twist` | `integrated_controller_node` → 底盤 | 最終速度命令 |

---

## 4. 各模組詳細說明

### 4.1 `mod_predict_yolo11_trt.py` — 道路檢測節點

**功能：**
1. 以 GStreamer Pipeline（Jetson 最佳化）或 V4L2 開啟 USB 相機（640×480 @ 30fps）
2. 載入 YOLO11n-seg TensorRT 引擎（`.engine` 格式），`max_det=1` 僅取最高信心度物件
3. 處理 letterbox padding，將 640×640 遮罩裁切並縮放回原始解析度
4. 萃取**雙參考點**：
   - **藍色前方參考點**（`forward_ref`）：沿畫面垂直中心線找遮罩最高點 → `e_d`（前方距離誤差）
   - **黃色橫向參考點**（`lateral_ref`）：固定在停止距離（1.1 m）對應像素高度，找該行遮罩中心 → `e_l`（橫向誤差）
5. 呼叫 `BEVTransformer.compute_errors()` 將像素座標轉為公尺誤差及變化率
6. 發布 `/road_info`；按 `s` 鍵切換避障開關，按 `q` 離開

**關鍵參數：**
```
confidence threshold : 0.5
IOU threshold        : 0.5
lateral_ref distance : 1.1 m（停止距離）
loop rate            : 30 Hz
```

---

### 4.2 `mod_fuzzy_control4.py` — 模糊控制節點

詳見 [第 5 章模糊控制器設計](#5-模糊控制器設計)。

**關鍵行為：**
- 訂閱 `/road_info`，當 `road_detected=1` 且系統啟用時執行模糊推論
- 推論結果加 EMA 低通濾波後，發布至 `/fuzzy_cmd_vel`
- 統計推論時間並記錄至 `outputs/fuzzy_stats_<timestamp>.txt`
- I/O 詳細資料記錄至 `outputs/fuzzy_io_<timestamp>.csv`
- 語音指令 `10000`（停止）可同時關閉鍵盤避障控制

---

### 4.3 `mod_voice_processing4.py` — 語音指令節點

**語音處理流程：**

```
麥克風 (PyAudio, 16kHz mono)
    │
    ▼ VAD (RMS 閾值偵測)
WAV 錄音（最長 5 秒）
    │
    ▼ noisereduce 降噪
WAV → MP3 (pydub)
    │
    ▼ base64 編碼
HTTPS POST → 遠端 LLM 語音服務
(ngrok tunnel, endpoint: /IAAgent/transcribe_translate)
    │
    ▼ JSON Response: {transcription, inst_codes[]}
    │
    ▼ 驗證 5 位數碼 + 冷卻時間 (1.0 s)
發布 /voice_command_code
```

**測試模式（`test_mode:=true`）：** 以鍵盤數字 0–9 模擬語音指令，無需麥克風與網路。

**VAD 參數：**
```
SILENCE_THRESHOLD       : 1000 (RMS，可由 ROS param ~silence_threshold 覆蓋)
SPEECH_START_THRESHOLD  : 5 chunks（連續 5 塊超閾值才開始錄音）
SPEECH_END_THRESHOLD    : 10 chunks（連續 10 塊靜音才停止）
MAX_RECORDING_SECONDS   : 5 s
```

---

### 4.4 `integrated_control4.py` — 整合控制節點

**三種運行狀態（主循環 10Hz）：**

| 狀態 | 條件 | 行為 |
|------|------|------|
| **完全暫停** | `paused=True` | 不輸出任何速度 |
| **語音指令執行中** | `is_executing_voice_cmd=True` & `cmd_executing=True` | 按指令類型執行（轉彎、前進、左右觀察）；超時 25 s 強制恢復 |
| **模糊控制模式** | `!paused` & `!is_executing_voice_cmd` & 避障啟用 | 直接轉發 `/fuzzy_cmd_vel` 至 `/cmd_vel` |

**指令優先序：**
`停止(10000)` > `開始任務(21000)` > `轉彎(13xxx/14xxx)` > `前進/避障(11xxx/15xxx/16xxx)` > 模糊控制

**轉彎後延遲機制：** `15000`/`16000` 指令完成後，延遲 1.5 秒再啟動模糊避障，避免轉彎瞬間誤判。

---

### 4.5 `fuzzy_data_logger.py` — 資料記錄節點

- 以 10 Hz 同步訂閱 `/road_info`、`/fuzzy_cmd_vel`、`/cmd_vel`、`/avoidance_enabled`、`/system_status`
- 寫入 CSV（`fuzzy_data_<timestamp>.csv`），欄位包含所有誤差輸入、模糊輸出、實際輸出
- 節點關閉時自動產生 `.summary.txt`，包含 `e_d`/`e_l` 範圍、`v`/`omega` 最大值及調參建議
- 預設輸出路徑：`src/fuzzy_control_design/logs/`

---

### 4.6 `resource_monitor_node.py` — 資源監控節點

- 使用 `tegrastats`（Jetson 原生工具）採樣系統資源，精確捕捉 GPU 記憶體與功耗
- 啟動前記錄**基準 RAM**，後續以差值計算 App RAM 增量
- 採樣間隔：`~interval`（預設 1.0 s）；`~ready_delay`（預設 5.0 s）後標記系統就緒，計算啟動延遲
- CSV 欄位：`Timestamp`, `CPU_Avg_Pct`, `CPU_Sum_Pct`, `GPU_Pct`, `RAM_Used_MB`, `App_RAM_Used_MB`, `Temp_GPU_C`, `Temp_CPU_C`
- 關閉時產生 `_report.txt`，包含 Peak/Avg CPU、RAM、GPU 統計

---

## 5. 模糊控制器設計

### 輸入變數（4個）

| 變數 | 範圍 | 語言值（5級）| 說明 |
|------|------|------------|------|
| `e_d` | `[0, 2.08]` m | VN / N / M / F / VF | 前方道路距離誤差 |
| `e_d_dot` | `[-2.08, 2.08]` m/s | NB / NS / ZO / PS / PB | `e_d` 變化率 |
| `e_l` | `[-0.5, 0.5]` m | NB / NS / ZO / PS / PB | 橫向位置誤差（含 ±0.08 m 死區）|
| `e_l_dot` | `[-1.0, 1.0]` m/s | NB / NS / ZO / PS / PB | `e_l` 變化率 |

### 輸出變數（2個，Singleton 解模糊化）

| 變數 | 語言值 | Singleton 值 |
|------|--------|-------------|
| `v`（線速度）| S / VS / SL / M / F | 0 / 0.18 / 0.32 / 0.45 / 0.55 m/s |
| `omega`（角速度）| NB / NS / ZO / PS / PB | -1.4 / -0.5 / 0 / 0.5 / 1.4 rad/s |

### 推論方法

- **規則數量**：625 條（`5^4`），從 CSV 檔載入（`src/fuzzy_control_design/fuzzy_rules_relaxed.csv`）
- **聚合（AND）**：Minimum（Mamdani 最小法）
- **解模糊化**：加權平均法（Weighted Average / Center of Gravity）
- **後處理**：EMA 低通濾波（`α_v=0.2`, `α_ω=0.3`）+ 範圍限制（`[0.0, 0.55]`, `[-1.4, 1.4]`）

```
μ_firing = min(μ_e_d, μ_e_d_dot, μ_e_l, μ_e_l_dot)   ← Mamdani AND

v = Σ(μ_firing_i × v_singleton_i) / Σ(μ_firing_i)     ← Weighted Average

v_filtered = α_v × v + (1-α_v) × v_prev               ← EMA filter
```

---

## 6. 語音指令編碼

格式：`[設備碼 1位][動作碼 1位][參數碼 3位]`（共 5 位數字）

| 設備碼 | 設備 | 常用指令 |
|--------|------|---------|
| `1` | 載具控制 | `10000`=停止、`11000`=前進/避障、`13000`=左轉90°、`14000`=右轉90°、`15000`=左轉後前進、`16000`=右轉後前進 |
| `2` | 攝影機 | `21000`=左右觀察（90°+180°+90°）|
| `3` | 主機功能 | 回報狀況、識別搜尋等 |
| `4` | 執行裝置 | 開門、關門等 |

### 整合控制器對語音指令的處理邏輯

```
11000 / 15xxx / 16xxx → avoid_obstacle = True  → 模糊控制接管
10000                 → 全系統停止
21000                 → 執行左右觀察（8.35 秒動作序列）
13xxx / 14xxx         → 執行定時轉彎
```

---

## 7. 安裝與依賴

### 系統需求

- **平台**：NVIDIA Jetson Orin NX（或相容 Jetson 系列）
- **OS**：Ubuntu 20.04 + JetPack 5.x
- **ROS**：Noetic
- **CUDA / TensorRT**：JetPack 內建

### Python 依賴

```bash
pip install ultralytics        # YOLO11（含 TensorRT 匯出支援）
pip install opencv-python-headless numpy
pip install pyaudio pydub soundfile noisereduce  # 語音處理
pip install requests                              # HTTP to LLM server
pip install psutil                                # 資源監控（可選）
```

### ROS 依賴套件

```bash
sudo apt install ros-noetic-geometry-msgs ros-noetic-std-msgs
```

### YOLO11 TRT 模型

預設路徑：
```
src/yolov7_ros/src/ultralytics/runs/segment/yolo11_seg_model12/weights/best.engine
```

若路徑不同，可透過 ROS launch 參數覆蓋：
```xml
<param name="weights" value="/your/path/to/model.engine"/>
```

---

## 8. 啟動方式

### 完整系統啟動

```bash
roslaunch yolov7_ros integrated_system.launch
```

### 選項參數

```bash
# 關閉資料記錄
roslaunch yolov7_ros integrated_system.launch enable_logger:=false

# 關閉資源監控
roslaunch yolov7_ros integrated_system.launch enable_resource_monitor:=false

# 同時關閉兩個可選節點
roslaunch yolov7_ros integrated_system.launch enable_logger:=false enable_resource_monitor:=false
```

### 鍵盤控制（即時）

在 `road_detection_node` 視窗中：
- `s` — 切換避障開關
- `q` — 退出節點

### 語音測試模式

語音節點預設 `test_mode:=false`（啟用語音）。若需鍵盤模擬：

```bash
# 在 launch 中修改：
# <param name="test_mode" value="true" />
```

鍵盤映射（測試模式）：

| 按鍵 | 指令碼 | 動作 |
|------|--------|------|
| `0` | `10000` | 停止 |
| `1` | `21000` | 開始任務（左右觀察）|
| `2` | `11000` | 前進/啟動避障 |
| `3` | `13000` | 左轉 90° |
| `4` | `14000` | 右轉 90° |
| `5` | `15000` | 左轉後前進 |
| `6` | `16000` | 右轉後前進 |

---

## 9. 輸出檔案說明

```
outputs/
├── fps_log_<timestamp>.txt          # YOLO 推論 FPS 記錄
├── fuzzy_stats_<timestamp>.txt      # 模糊推論時間統計
├── fuzzy_io_<timestamp>.csv         # 模糊 I/O 詳細記錄（e_d, e_l, v, omega）
├── masks_<timestamp>/               # 分割遮罩圖（save_mask=true 時）
│   └── mask_000001.png
├── detection_<timestamp>.avi        # 偵測結果影片（save_video=true 時）
└── resource_logs/
    └── <YYYYMMDD>/
        ├── resources_<timestamp>.csv
        └── resources_<timestamp>_report.txt

src/fuzzy_control_design/logs/
└── fuzzy_data_<timestamp>.csv       # 完整資料記錄（fuzzy_data_logger）
    fuzzy_data_<timestamp>.summary.txt
```

---

## 10. 調參指引

### 模糊控制器調參

| 調整目標 | 建議操作 |
|---------|---------|
| 轉彎響應過強（擺動） | 降低 `OMEGA.NS/PS`（目前 ±0.5）|
| 轉彎響應不足（左右偏） | 提高 `OMEGA.NB/PB`（目前 ±1.4）|
| 速度過快 | 降低 `V.F`（目前 0.55 m/s）|
| 低速時仍抖動 | 降低 `alpha_omega`（目前 0.3）|
| e_l 穩態誤差大 | 縮小 `E_L.ZO` 死區（目前 ±0.08 m）|

### 語音靈敏度調整

```bash
# 增大閾值 → 更不靈敏（適合噪音環境）
rosparam set /voice_command_node/silence_threshold 1500

# 啟用音量偵錯
rosparam set /voice_command_node/debug_volume true
```

### 相機調整

```bash
# 透過 launch 參數調整解析度與幀率
roslaunch yolov7_ros integrated_system.launch
# 在節點參數中設定 camera_width, camera_height, use_gstreamer
```

---

## 授權 / License

本專案為學術研究用途（國科會 NSTC 計畫），程式碼僅供參考。  
YOLO11 模型遵循 [Ultralytics AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 授權。
