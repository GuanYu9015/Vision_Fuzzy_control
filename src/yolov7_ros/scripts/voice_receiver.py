#!/usr/bin/python3
"""
語音接收模組 (VoiceReceiver)

設計目的：
    提供可重用的語音接收功能，可獨立於 ROS 節點使用。
    支援實際錄音模式與測試模式（模擬輸入）。

使用方式：
    1. 獨立使用：python3 voice_receiver.py
    2. 模組導入：from voice_receiver import VoiceReceiver

主要功能：
    - 使用 VAD（語音活動檢測）進行智能錄音
    - 自動降噪處理
    - 雲端語音辨識（支援 AMRR 語音服務）
    - 指令編碼解析與驗證

作者：自動生成
版本：1.0.0
"""

import os
import json
import requests
import base64
import time
import io
import datetime
import shutil
from typing import Optional, Tuple, List, Dict, Callable, Any
from dataclasses import dataclass

import numpy as np

# 可選依賴：錄音相關
try:
    import pyaudio
    import wave
    from pydub import AudioSegment
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# 可選依賴：降噪相關
try:
    import soundfile as sf
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False


@dataclass
class VoiceConfig:
    """語音服務設定資料類別"""
    host: str = "prosaically-unpulvinate-donette.ngrok-free.dev"
    port: int = 443
    service_id: int = 22
    timeout: int = 45  # 請求超時時間（秒）


@dataclass
class AudioConfig:
    """錄音參數設定資料類別"""
    chunk_size: int = 1024           # 每次讀取的音頻塊大小
    format: int = 8                  # pyaudio.paInt16 的數值
    channels: int = 1                # 單聲道
    rate: int = 16000                # 取樣率 (Hz)
    silence_threshold: int = 1000    # 靜音閾值 (RMS)
    speech_start_frames: int = 5     # 開始說話的連續幀數
    speech_end_frames: int = 10      # 結束說話的靜音幀數
    max_duration: float = 5.0        # 最長錄音時間（秒）
    pre_buffer_seconds: float = 0.5  # 預緩衝時間（秒）


@dataclass
class RecognitionResult:
    """語音辨識結果資料類別"""
    success: bool                    # 辨識是否成功
    transcription: str               # 辨識文字
    inst_codes: List[str]            # 指令編碼列表
    raw_response: Dict               # 原始回應
    error_message: Optional[str] = None  # 錯誤訊息（如有）


# ===== 指令編碼定義 (NSTC 114.12.19 版) =====
# 編碼格式: [設備碼 1位][動作碼 1位][參數碼 3位]

VEHICLE_ACTIONS = {
    '0': '暫停待命', '1': '開始前進', '2': '繼續巡邏',
    '3': '左轉', '4': '右轉', '5': '左轉往前', '6': '右轉往前', '7': '播放影片'
}

VEHICLE_SPECIAL_CODES = {'21000': '左右觀察'}

CAMERA_ACTIONS = {
    '2': '上下觀察', '3': '鏡頭往左', '4': '鏡頭往右',
    '5': '鏡頭往上', '6': '鏡頭往下'
}

HOST_ACTIONS = {
    '1': '回報狀況', '2': '識別搜尋', '3': '測量',
    '4': '通知預約', '5': '紀錄'
}

ACTUATOR_ACTIONS = {
    '1': '開門', '2': '關門', '3': '解鎖', '4': '上鎖', '5': '拿給我'
}

LOCATION_CODES = {
    '01': '傷患', '02': '斜坡', '03': '門口', '04': '轉角處',
    '05': '廁所', '06': '電梯', '07': '樓梯', '08': '病房',
    '09': '護理站', '10': '急診室', '11': 'X光室'
}

VIDEO_CODES = {'01': '就診流程', '02': '火災疏散'}

ITEM_CODES = {
    '01': '水', '02': '咖啡', '03': '茶', '04': '汽水',
    '05': '碗', '06': '杯子', '07': '巧克力', '08': '蘋果', '09': '優格'
}


class VoiceReceiver:
    """
    語音接收器類別
    
    提供獨立於 ROS 的語音錄音、辨識和指令解析功能。
    
    屬性：
        test_mode (bool): 是否為測試模式
        recordings_dir (str): 錄音檔案儲存目錄
        voice_config (VoiceConfig): 語音服務設定
        audio_config (AudioConfig): 錄音參數設定
    
    範例用法：
        >>> receiver = VoiceReceiver(test_mode=False)
        >>> result = receiver.listen_and_recognize()
        >>> if result.success:
        >>>     print(f"辨識結果: {result.transcription}")
        >>>     for code in result.inst_codes:
        >>>         print(f"指令: {code}")
    """
    
    def __init__(
        self,
        test_mode: bool = True,
        recordings_dir: Optional[str] = None,
        voice_config: Optional[VoiceConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        logger: Optional[Callable[[str], None]] = None
    ):
        """
        初始化語音接收器
        
        參數：
            test_mode: 是否使用測試模式（不實際錄音）
            recordings_dir: 錄音檔案儲存目錄，預設為 ./recordings
            voice_config: 語音服務設定，使用預設值若為 None
            audio_config: 錄音參數設定，使用預設值若為 None
            logger: 自訂日誌函數，預設為 print
        """
        self.test_mode = test_mode
        self.voice_config = voice_config or VoiceConfig()
        self.audio_config = audio_config or AudioConfig()
        self.logger = logger or print
        
        # 設定錄音目錄
        if recordings_dir is None:
            self.recordings_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'recordings'
            )
        else:
            self.recordings_dir = recordings_dir
        
        self._ensure_directory(self.recordings_dir)
        
        # 暫存檔路徑
        self.temp_wav_path = os.path.join(self.recordings_dir, 'temp_recording.wav')
        
        # 初始化音頻系統（非測試模式）
        self.pyaudio_instance = None
        self.input_device_index = None
        if not self.test_mode and AUDIO_AVAILABLE:
            self._init_audio_system()
        elif not AUDIO_AVAILABLE:
            self._log("警告：pyaudio/pydub 未安裝，強制進入測試模式")
            self.test_mode = True
        
        self._log(f"VoiceReceiver 初始化完成 (測試模式: {self.test_mode})")
    
    def _log(self, message: str) -> None:
        """輸出日誌訊息"""
        self.logger(f"[VoiceReceiver] {message}")
    
    def _ensure_directory(self, path: str) -> bool:
        """確保目錄存在"""
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                self._log(f"創建目錄: {path}")
                return True
            except OSError as e:
                self._log(f"無法創建目錄 {path}: {e}")
                return False
        return True
    
    def _init_audio_system(self) -> None:
        """初始化音頻系統"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.input_device_index = self._find_input_device()
            info = self.pyaudio_instance.get_device_info_by_index(self.input_device_index)
            self._log(f"輸入裝置: {info['name']}")
        except Exception as e:
            self._log(f"PyAudio 初始化失敗: {e}")
            self.test_mode = True
    
    def _find_input_device(self, keywords: Tuple[str, ...] = ('pulse', 'bluetooth')) -> int:
        """自動搜尋輸入裝置"""
        if self.pyaudio_instance is None:
            raise RuntimeError("PyAudio instance not initialized")
        
        for i in range(self.pyaudio_instance.get_device_count()):
            info = self.pyaudio_instance.get_device_info_by_index(i)
            name = info['name'].lower()
            if info['maxInputChannels'] > 0 and any(kw in name for kw in keywords):
                return i
        return self.pyaudio_instance.get_default_input_device_info()['index']
    
    def calculate_rms(self, audio_data: bytes) -> float:
        """
        計算音頻塊的均方根值 (RMS)，作為音量大小的指標
        
        參數：
            audio_data: 音頻位元組資料
        
        回傳：
            RMS 值（浮點數）
        """
        try:
            shorts = np.frombuffer(audio_data, dtype=np.int16)
            if len(shorts) == 0:
                return 0.0
            squares = np.square(shorts.astype(np.float64))
            mean_squares = np.mean(squares)
            if mean_squares <= 0:
                return 0.0
            return np.sqrt(mean_squares)
        except Exception as e:
            self._log(f"計算 RMS 時出錯: {e}")
            return 0.0
    
    def record_audio_with_vad(self, stop_callback: Optional[Callable[[], bool]] = None) -> Optional[str]:
        """
        使用 VAD 算法進行智能錄音
        
        參數：
            stop_callback: 可選的停止回調函數，回傳 True 時停止錄音
        
        回傳：
            錄音檔案路徑，或 None（錄音失敗/無有效語音）
        """
        if self.test_mode:
            self._log("測試模式：跳過實際錄音")
            return self.temp_wav_path
        
        if self.pyaudio_instance is None:
            self._log("錯誤：PyAudio 未初始化")
            return None
        
        cfg = self.audio_config
        pre_buffer_size = int(cfg.pre_buffer_seconds * cfg.rate / cfg.chunk_size)
        
        try:
            stream = self.pyaudio_instance.open(
                format=cfg.format,
                channels=cfg.channels,
                rate=cfg.rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=cfg.chunk_size
            )
            
            frames = []
            buffer = []  # 預緩衝
            silent_chunks = 0
            voiced_chunks = 0
            recording = False
            start_time = time.time()
            last_log_time = time.time()
            
            self._log("開始監聽語音...")
            
            while True:
                # 檢查停止條件
                if stop_callback and stop_callback():
                    self._log("收到停止請求")
                    break
                
                # 檢查最長錄音時間
                if recording and (time.time() - start_time > cfg.max_duration):
                    self._log(f"達到最大錄音時間 ({cfg.max_duration}秒)")
                    break
                
                data = stream.read(cfg.chunk_size, exception_on_overflow=False)
                current_rms = self.calculate_rms(data)
                
                # 每 2 秒輸出一次音量
                if time.time() - last_log_time >= 2.0:
                    self._log(f"即時音量 (RMS): {current_rms:.2f}")
                    last_log_time = time.time()
                
                # 未開始錄製時維護預緩衝
                if not recording:
                    buffer.append(data)
                    if len(buffer) > pre_buffer_size:
                        buffer.pop(0)
                
                # 音量檢測邏輯
                if current_rms > cfg.silence_threshold:
                    if not recording:
                        voiced_chunks += 1
                        if voiced_chunks >= cfg.speech_start_frames:
                            recording = True
                            frames.extend(buffer)
                            self._log("檢測到語音，開始錄音...")
                            start_time = time.time()
                    else:
                        silent_chunks = 0
                        frames.append(data)
                else:
                    if recording:
                        silent_chunks += 1
                        frames.append(data)
                        if silent_chunks >= cfg.speech_end_frames:
                            self._log("檢測到語音結束，停止錄音...")
                            break
                    else:
                        voiced_chunks = 0
            
            stream.stop_stream()
            stream.close()
            
            # 檢查是否錄到有效語音
            if not recording or len(frames) < cfg.speech_start_frames:
                self._log("未檢測到有效語音")
                return None
            
            # 儲存 WAV 檔案
            wf = wave.open(self.temp_wav_path, 'wb')
            wf.setnchannels(cfg.channels)
            wf.setsampwidth(self.pyaudio_instance.get_sample_size(cfg.format))
            wf.setframerate(cfg.rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            self._log(f"錄音已保存: {self.temp_wav_path}")
            return self.temp_wav_path
            
        except Exception as e:
            self._log(f"錄音過程中發生錯誤: {e}")
            return None
    
    def apply_noise_reduction(self, wav_path: str) -> str:
        """
        對音頻檔案進行降噪處理
        
        參數：
            wav_path: 輸入 WAV 檔案路徑
        
        回傳：
            處理後的檔案路徑（可能與輸入相同，若降噪失敗）
        """
        if not NOISE_REDUCE_AVAILABLE:
            self._log("降噪庫未安裝，跳過降噪處理")
            return wav_path
        
        try:
            self._log("開始進行降噪處理...")
            y, sr = sf.read(wav_path)
            noise_clip = y[:int(0.5 * sr)]  # 取前 0.5 秒作為噪聲樣本
            reduced = nr.reduce_noise(y, sr, noise_clip)
            
            reduced_path = os.path.join(self.recordings_dir, "reduced_recording.wav")
            sf.write(reduced_path, reduced, sr)
            self._log(f"降噪完成: {reduced_path}")
            return reduced_path
        except Exception as e:
            self._log(f"降噪失敗，保留原始錄音: {e}")
            return wav_path
    
    def convert_to_mp3(self, wav_path: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        將 WAV 檔案轉換為 MP3 格式
        
        參數：
            wav_path: WAV 檔案路徑
        
        回傳：
            (MP3 位元組資料, MP3 檔案路徑) 或 (None, None) 若失敗
        """
        if not AUDIO_AVAILABLE:
            self._log("pydub 未安裝，無法轉換 MP3")
            return None, None
        
        try:
            sound = AudioSegment.from_wav(wav_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mp3_path = os.path.join(self.recordings_dir, f"recording_{timestamp}.mp3")
            
            mp3_io = io.BytesIO()
            sound.export(mp3_io, format="mp3")
            mp3_bytes = mp3_io.getvalue()
            
            return mp3_bytes, mp3_path
        except Exception as e:
            self._log(f"轉換 MP3 時出錯: {e}")
            return None, None
    
    def get_audio_base64(self, apply_noise_reduce: bool = True) -> Optional[str]:
        """
        錄音並轉換為 Base64 編碼
        
        參數：
            apply_noise_reduce: 是否進行降噪處理
        
        回傳：
            Base64 編碼字串，或 None（失敗）
        """
        if self.test_mode:
            return ""  # 測試模式回傳空字串
        
        wav_path = self.record_audio_with_vad()
        if not wav_path:
            return None
        
        # 降噪處理
        if apply_noise_reduce:
            wav_path = self.apply_noise_reduction(wav_path)
        
        # 轉換為 MP3
        mp3_bytes, mp3_path = self.convert_to_mp3(wav_path)
        if not mp3_bytes:
            return None
        
        # 儲存檔案副本
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_copy = os.path.join(self.recordings_dir, f"recording_{timestamp}.wav")
            shutil.copy2(wav_path, wav_copy)
            
            if mp3_path:
                with open(mp3_path, 'wb') as f:
                    f.write(mp3_bytes)
                self._log(f"MP3 已保存: {mp3_path}")
        except Exception as e:
            self._log(f"保存檔案時出錯: {e}")
        
        # 清理暫存檔
        for temp_file in [self.temp_wav_path, 
                          os.path.join(self.recordings_dir, "reduced_recording.wav")]:
            if os.path.exists(temp_file) and temp_file != wav_path:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
        
        return base64.b64encode(mp3_bytes).decode('utf-8')
    
    def send_recognition_request(self, audio_base64: str) -> Dict[str, Any]:
        """
        發送語音辨識請求到雲端服務
        
        參數：
            audio_base64: Base64 編碼的音頻資料
        
        回傳：
            API 回應的 JSON 字典
        """
        if self.test_mode or not audio_base64:
            return {"response": {"inst_codes": []}}
        
        cfg = self.voice_config
        
        # 建構 URL
        if cfg.port and cfg.port != 443:
            api_url = f'https://{cfg.host}:{cfg.port}/IAAgent/transcribe_translate'
        else:
            api_url = f'https://{cfg.host}/IAAgent/transcribe_translate'
        
        try:
            response = requests.post(
                api_url,
                json={
                    "serviceid": cfg.service_id,
                    "func": "amrr-001",
                    "conn": "ntnu-12345",
                    "invoice": audio_base64
                },
                timeout=cfg.timeout
            )
            
            if response.status_code != 200:
                self._log(f"語音服務返回非 200 狀態碼: {response.status_code}")
                return {"response": {}}
            
            return response.json()
        except requests.exceptions.RequestException as e:
            self._log(f"發送語音請求時出錯: {e}")
            return {"response": {}}
        except json.JSONDecodeError as e:
            self._log(f"解析 JSON 回應時出錯: {e}")
            return {"response": {}}
    
    def is_valid_code(self, code: str) -> bool:
        """
        驗證指令編碼是否有效
        
        參數：
            code: 5 位數指令編碼
        
        回傳：
            是否為有效編碼
        """
        if len(code) != 5:
            return False
        
        device_code = code[0]
        action_code = code[1]
        
        # 設備 1：載具控制
        if device_code == '1':
            return action_code in VEHICLE_ACTIONS
        
        # 設備 2：攝影機 (含 21000 左右觀察)
        if device_code == '2':
            if code in VEHICLE_SPECIAL_CODES:
                return True
            return action_code in CAMERA_ACTIONS
        
        # 設備 3：主機功能
        if device_code == '3':
            return action_code in HOST_ACTIONS
        
        # 設備 4：執行裝置
        if device_code == '4':
            return action_code in ACTUATOR_ACTIONS
        
        return False
    
    def parse_code(self, code: str) -> Dict[str, str]:
        """
        解析指令編碼
        
        參數：
            code: 5 位數指令編碼
        
        回傳：
            包含 device, action, param, description 的字典
        """
        if not self.is_valid_code(code):
            return {"error": f"無效編碼: {code}"}
        
        device_code = code[0]
        action_code = code[1]
        param_code = code[2:5]
        
        result = {
            "code": code,
            "device_code": device_code,
            "action_code": action_code,
            "param_code": param_code
        }
        
        # 特殊處理 21000
        if code in VEHICLE_SPECIAL_CODES:
            result["device"] = "載具"
            result["action"] = VEHICLE_SPECIAL_CODES[code]
            result["description"] = f"載具：{VEHICLE_SPECIAL_CODES[code]}"
            return result
        
        device_map = {
            '1': ('載具', VEHICLE_ACTIONS),
            '2': ('攝影機', CAMERA_ACTIONS),
            '3': ('主機', HOST_ACTIONS),
            '4': ('執行裝置', ACTUATOR_ACTIONS)
        }
        
        if device_code in device_map:
            device_name, actions = device_map[device_code]
            action_name = actions.get(action_code, '未知動作')
            result["device"] = device_name
            result["action"] = action_name
            result["description"] = f"{device_name}：{action_name} (參數: {param_code})"
        
        return result
    
    def listen_and_recognize(self, apply_noise_reduce: bool = True) -> RecognitionResult:
        """
        執行完整的語音接收與辨識流程
        
        參數：
            apply_noise_reduce: 是否進行降噪處理
        
        回傳：
            RecognitionResult 資料類別
        """
        # 取得音頻 Base64
        audio_base64 = self.get_audio_base64(apply_noise_reduce)
        
        if audio_base64 is None:
            return RecognitionResult(
                success=False,
                transcription="",
                inst_codes=[],
                raw_response={},
                error_message="錄音失敗或無法取得音頻資料"
            )
        
        # 發送辨識請求
        response = self.send_recognition_request(audio_base64)
        
        # 解析回應
        resp_data = response.get('response', {})
        transcription = resp_data.get('transcription', '')
        inst_codes = resp_data.get('inst_codes', [])
        
        # 轉換為字串並填充至 5 位
        inst_codes = [str(c).zfill(5) for c in inst_codes]
        
        # 過濾有效編碼
        valid_codes = [c for c in inst_codes if self.is_valid_code(c)]
        
        return RecognitionResult(
            success=True,
            transcription=transcription,
            inst_codes=valid_codes,
            raw_response=response
        )
    
    def simulate_input(self, key: str) -> Optional[str]:
        """
        模擬語音輸入（測試用）
        
        參數：
            key: 按鍵字元 (0-9)
        
        回傳：
            對應的指令編碼，或 None（無效輸入）
        """
        code_map = {
            "0": "10000",  # 停止
            "1": "21000",  # 開始任務/左右觀察
            "2": "11000",  # 前進
            "3": "13000",  # 左轉90度
            "4": "14000",  # 右轉90度
            "5": "15000",  # 左轉後前進
            "6": "16000",  # 右轉後前進
            "7": "11110",  # 前進10公尺
            "8": "13130",  # 左轉30度
            "9": "14145",  # 右轉45度
        }
        
        return code_map.get(key)
    
    def cleanup(self) -> None:
        """清理資源"""
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
                self._log("PyAudio 已終止")
            except Exception as e:
                self._log(f"PyAudio 終止失敗: {e}")
    
    def __del__(self):
        """解構時自動清理"""
        self.cleanup()


# ===== 獨立執行範例 =====
def main():
    """獨立執行範例"""
    print("=" * 60)
    print("VoiceReceiver 獨立測試")
    print("=" * 60)
    
    # 建立接收器（測試模式）
    receiver = VoiceReceiver(test_mode=True)
    
    print("\n測試模式：輸入 0-9 模擬語音指令，輸入 q 離開\n")
    print("按鍵對照：")
    print("  0 = 停止        1 = 開始任務    2 = 前進")
    print("  3 = 左轉90°     4 = 右轉90°     5 = 左轉後前進")
    print("  6 = 右轉後前進   7 = 前進10m     8 = 左轉30°")
    print("  9 = 右轉45°\n")
    
    while True:
        try:
            key = input("請輸入指令 (0-9, q=離開): ").strip()
            
            if key.lower() == 'q':
                print("結束測試")
                break
            
            code = receiver.simulate_input(key)
            if code:
                parsed = receiver.parse_code(code)
                print(f"  編碼: {code}")
                print(f"  解析: {parsed.get('description', '無效編碼')}")
                print(f"  有效: {receiver.is_valid_code(code)}")
                print()
            else:
                print("  無效輸入\n")
                
        except KeyboardInterrupt:
            print("\n使用者中斷")
            break
        except EOFError:
            break
    
    receiver.cleanup()


if __name__ == "__main__":
    main()
