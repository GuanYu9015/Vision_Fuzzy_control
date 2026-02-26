# VoiceReceiver 語音接收模組使用說明

## 簡介

`VoiceReceiver` 是從 `mod_voice_processing4.py` 提取的**獨立語音接收模組**，可在任何 Python 程式中使用，不依賴 ROS 系統。

---

## 主要功能

| 功能 | 說明 |
|------|------|
| **VAD 智能錄音** | 使用語音活動檢測 (Voice Activity Detection) 自動開始/結束錄音 |
| **自動降噪** | 使用 `noisereduce` 庫進行噪聲消除 |
| **雲端辨識** | 連接 AMRR 語音服務進行語音轉文字 |
| **指令解析** | 支援 NSTC 114.12.19 版指令編碼系統 |
| **測試模式** | 不需麥克風即可測試指令流程 |

---

## 快速開始

### 1. 安裝依賴

```bash
# 核心依賴
pip install numpy requests

# 錄音功能（可選，測試模式不需要）
pip install pyaudio pydub

# 降噪功能（可選）
pip install soundfile noisereduce
```

> **注意**：PyAudio 在 Linux 上可能需要先安裝系統套件：
> ```bash
> sudo apt-get install portaudio19-dev python3-pyaudio
> ```

### 2. 基本使用

```python
from voice_receiver import VoiceReceiver, RecognitionResult

# 建立接收器（預設為測試模式）
receiver = VoiceReceiver(test_mode=False)

# 執行語音接收與辨識
result = receiver.listen_and_recognize()

if result.success:
    print(f"辨識文字: {result.transcription}")
    for code in result.inst_codes:
        print(f"指令編碼: {code}")
        parsed = receiver.parse_code(code)
        print(f"  說明: {parsed['description']}")

# 記得清理資源
receiver.cleanup()
```

---

## API 參考

### VoiceReceiver 類別

#### 初始化參數

```python
VoiceReceiver(
    test_mode: bool = True,           # 測試模式（不實際錄音）
    recordings_dir: str = None,       # 錄音儲存目錄
    voice_config: VoiceConfig = None, # 語音服務設定
    audio_config: AudioConfig = None, # 錄音參數設定
    logger: Callable = None           # 自訂日誌函數
)
```

#### 主要方法

| 方法 | 說明 | 回傳值 |
|------|------|--------|
| `listen_and_recognize()` | 完整語音接收與辨識流程 | `RecognitionResult` |
| `record_audio_with_vad()` | 使用 VAD 錄音 | `str` (檔案路徑) 或 `None` |
| `apply_noise_reduction(wav_path)` | 降噪處理 | `str` (處理後路徑) |
| `send_recognition_request(base64)` | 發送辨識請求 | `dict` |
| `is_valid_code(code)` | 驗證指令編碼 | `bool` |
| `parse_code(code)` | 解析指令編碼 | `dict` |
| `simulate_input(key)` | 模擬鍵盤輸入 | `str` 或 `None` |
| `cleanup()` | 清理資源 | `None` |

### 設定類別

#### VoiceConfig（語音服務設定）

```python
@dataclass
class VoiceConfig:
    host: str = "prosaically-unpulvinate-donette.ngrok-free.dev"
    port: int = 443
    service_id: int = 22
    timeout: int = 45
```

#### AudioConfig（錄音參數設定）

```python
@dataclass
class AudioConfig:
    chunk_size: int = 1024           # 音頻塊大小
    format: int = 8                  # paInt16
    channels: int = 1                # 單聲道
    rate: int = 16000                # 取樣率
    silence_threshold: int = 1000    # 靜音閾值
    speech_start_frames: int = 5     # 開始說話判定幀數
    speech_end_frames: int = 10      # 結束說話判定幀數
    max_duration: float = 5.0        # 最長錄音秒數
    pre_buffer_seconds: float = 0.5  # 預緩衝秒數
```

### RecognitionResult（辨識結果）

```python
@dataclass
class RecognitionResult:
    success: bool              # 是否成功
    transcription: str         # 辨識文字
    inst_codes: List[str]      # 指令編碼列表
    raw_response: Dict         # 原始回應
    error_message: str = None  # 錯誤訊息
```

---

## 指令編碼系統

### 編碼格式

```
[設備碼 1位][動作碼 1位][參數碼 3位]
```

### 設備碼對照

| 設備碼 | 設備類型 | 說明 |
|--------|----------|------|
| 1 | 載具控制 | 機器人移動控制 |
| 2 | 攝影機 | 雲台/鏡頭控制 |
| 3 | 主機功能 | 辨識、回報等 |
| 4 | 執行裝置 | 開門、解鎖等 |

### 載具指令（設備碼 1）

| 編碼 | 動作 | 說明 |
|------|------|------|
| 10000 | 暫停待命 | 停止所有動作 |
| 11000 | 開始前進 | 啟動避障前進 |
| 13000 | 左轉 | 左轉 90° |
| 14000 | 右轉 | 右轉 90° |
| 15000 | 左轉往前 | 左轉後前進 |
| 16000 | 右轉往前 | 右轉後前進 |
| 131XX | 左轉 XX° | 例：13130 = 左轉30° |
| 141XX | 右轉 XX° | 例：14145 = 右轉45° |

### 特殊指令

| 編碼 | 動作 | 說明 |
|------|------|------|
| 21000 | 左右觀察 | 載具執行左右轉90°觀察 |

---

## 使用範例

### 範例 1：基本語音接收

```python
from voice_receiver import VoiceReceiver

receiver = VoiceReceiver(test_mode=False)

try:
    result = receiver.listen_and_recognize()
    if result.success and result.inst_codes:
        print(f"收到指令: {result.inst_codes}")
finally:
    receiver.cleanup()
```

### 範例 2：自訂設定

```python
from voice_receiver import VoiceReceiver, VoiceConfig, AudioConfig

# 自訂語音服務
voice_cfg = VoiceConfig(
    host="your-server.example.com",
    port=8080,
    timeout=30
)

# 調整錄音參數（例如：降低靜音閾值）
audio_cfg = AudioConfig(
    silence_threshold=500,
    max_duration=10.0
)

receiver = VoiceReceiver(
    test_mode=False,
    voice_config=voice_cfg,
    audio_config=audio_cfg
)
```

### 範例 3：整合到 ROS 節點

```python
import rospy
from std_msgs.msg import String
from voice_receiver import VoiceReceiver

class MyVoiceNode:
    def __init__(self):
        rospy.init_node('my_voice_node')
        self.pub = rospy.Publisher('/voice_cmd', String, queue_size=10)
        
        # 使用 ROS logger
        self.receiver = VoiceReceiver(
            test_mode=rospy.get_param('~test_mode', False),
            logger=rospy.loginfo
        )
    
    def run(self):
        while not rospy.is_shutdown():
            result = self.receiver.listen_and_recognize()
            if result.success:
                for code in result.inst_codes:
                    self.pub.publish(code)
            rospy.sleep(0.1)
```

### 範例 4：測試模式

```python
from voice_receiver import VoiceReceiver

receiver = VoiceReceiver(test_mode=True)

# 模擬使用者輸入「2」(前進)
code = receiver.simulate_input("2")
print(f"模擬指令: {code}")  # 輸出: 11000

# 解析指令
parsed = receiver.parse_code(code)
print(f"說明: {parsed['description']}")
# 輸出: 載具：開始前進 (參數: 000)
```

### 範例 5：只使用指令解析

```python
from voice_receiver import VoiceReceiver

receiver = VoiceReceiver(test_mode=True)

# 驗證編碼
print(receiver.is_valid_code("13045"))  # True
print(receiver.is_valid_code("99999"))  # False

# 解析編碼
info = receiver.parse_code("15000")
# {'code': '15000', 'device_code': '1', 'action_code': '5', 
#  'param_code': '000', 'device': '載具', 'action': '左轉往前', 
#  'description': '載具：左轉往前 (參數: 000)'}
```

---

## 與原始 mod_voice_processing4.py 的差異

| 項目 | mod_voice_processing4.py | voice_receiver.py |
|------|--------------------------|-------------------|
| ROS 依賴 | 必須 | 可選 |
| 執行方式 | ROS 節點 | 獨立模組 |
| 設定方式 | ROS 參數 | 建構函數參數 |
| 日誌輸出 | rospy.loginfo | 自訂或 print |
| 狀態發布 | ROS Topic | 回傳值 |

---

## 故障排除

### 問題：pyaudio 安裝失敗

```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev
pip install pyaudio

# 或使用 conda
conda install pyaudio
```

### 問題：找不到輸入裝置

模組會自動搜尋包含 `pulse` 或 `bluetooth` 的裝置。若需指定其他裝置，可覆寫 `_find_input_device` 方法。

### 問題：辨識結果為空

1. 檢查網路連線
2. 確認語音服務設定正確
3. 調整 `silence_threshold` 參數（環境噪音較大時需提高）

---

## 檔案結構

```
scripts/
├── voice_receiver.py          # 可重用語音模組
├── VOICE_RECEIVER_README.md   # 本說明文件
├── mod_voice_processing4.py   # 原始 ROS 節點（參考）
└── recordings/                # 錄音儲存目錄（自動建立）
```

---

## 授權資訊

此模組基於專案現有程式碼設計，供專案內部使用。
