#!/usr/bin/python3
"""
語音指令處理節點
功能：錄音、降噪、語音辨識、指令發布
"""

import os
import json
import requests
import base64
import time
import threading
import io
import sys
import datetime
import shutil

import numpy as np
import pyaudio
import wave
from pydub import AudioSegment

import rospy
import rospkg
from std_msgs.msg import String, Bool

# === 語音服務設定 (可透過 ROS 參數覆蓋) ===
AMRR_VOICE_SERVICE_HOST = "prosaically-unpulvinate-donette.ngrok-free.dev"
AMRR_VOICE_SERVICE_PORT = 443
AMRR_VOICE_SERVICE_ID = 22

# ===== 指令編碼定義 (NSTC 114.12.19 版) =====
# 編碼格式: [設備碼 1位][動作碼 1位][參數碼 3位]

# 設備 1：載具控制
VEHICLE_ACTIONS = {
    '0': '暫停待命', '1': '開始前進', '2': '繼續巡邏',
    '3': '左轉', '4': '右轉', '5': '左轉往前', '6': '右轉往前', '7': '播放影片'
}
# 特殊載具指令 (設備碼 2 但實際由載具執行)
VEHICLE_SPECIAL_CODES = {'21000': '左右觀察'}  # 載具左右轉90°實現

# 設備 2：攝影機觀察
CAMERA_ACTIONS = {
    '2': '上下觀察', '3': '鏡頭往左', '4': '鏡頭往右',
    '5': '鏡頭往上', '6': '鏡頭往下'
}

# 設備 3：主機功能與辨識
HOST_ACTIONS = {
    '1': '回報狀況', '2': '識別搜尋', '3': '測量',
    '4': '通知預約', '5': '紀錄'
}

# 設備 4：執行裝置
ACTUATOR_ACTIONS = {
    '1': '開門', '2': '關門', '3': '解鎖', '4': '上鎖', '5': '拿給我'
}

# 參數代碼表
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

# 有效指令前綴 (向後相容)
VALID_MOVEMENT_PREFIXES = ('11', '15', '16')
VALID_TURN_PREFIXES = ('13', '14')


class VoiceCommandNode:
    """語音指令處理節點"""
    
    def __init__(self):
        rospy.init_node('voice_command_node', anonymous=True)
        
        # 使用 rospkg 取得套件路徑
        rospack = rospkg.RosPack()
        try:
            self.pkg_path = rospack.get_path('yolov7_ros')
        except rospkg.ResourceNotFound:
            rospy.logwarn("Package 'yolov7_ros' not found, using current directory")
            self.pkg_path = os.path.dirname(os.path.abspath(__file__))
        
        # 發布者與訂閱者
        self.voice_pub = rospy.Publisher('/voice_command_code', String, queue_size=10)
        self.system_status_pub = rospy.Publisher('/system_status', String, queue_size=10)
        rospy.Subscriber('/system_status', String, self.system_status_callback)
        
        # 狀態追蹤
        self.task_started = False
        self.avoid_obstacle = False
        
        # 測試模式設定
        self.test_mode = rospy.get_param('~test_mode', True)
        rospy.loginfo(f"測試模式: {self.test_mode}")
        
        # 錄音目錄設定 (使用相對路徑)
        default_recordings_dir = os.path.join(self.pkg_path, 'recordings')
        self.recordings_dir = rospy.get_param('~recordings_dir', default_recordings_dir)
        self._ensure_directory(self.recordings_dir)
        
        # 初始化音頻系統
        self._init_audio_system()
        
        # 錄音暫存檔
        self.wav_filename = os.path.join(self.recordings_dir, 'temp_recording.wav')
        
        # 控制旗標
        self.show_prompt = True
        self.is_processing = False
        self.stop_requested = False
        self.last_cmd_time = 0
        self.cmd_cooldown = 1.0  # 指令冷卻時間（秒）
        
        self._log_test_mode_help()
        rospy.loginfo("語音指令節點已初始化")
    
    def _ensure_directory(self, path):
        """確保目錄存在"""
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                rospy.loginfo(f"創建目錄: {path}")
            except OSError as e:
                rospy.logerr(f"無法創建目錄 {path}: {e}")
                return False
        return True
    
    def _init_audio_system(self):
        """初始化音頻系統"""
        if self.test_mode:
            return
        
        # 錄音參數
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        # VAD 參數
        self.SILENCE_THRESHOLD = rospy.get_param('~silence_threshold', 1000)
        self.SPEECH_START_THRESHOLD = 5
        self.SPEECH_END_THRESHOLD = 10
        self.MAX_RECORDING_SECONDS = 5
        
        rospy.loginfo(f"靜音閾值: {self.SILENCE_THRESHOLD}")
        
        try:
            self.p = pyaudio.PyAudio()
            self.input_device_index = self._find_input_device()
            info = self.p.get_device_info_by_index(self.input_device_index)
            rospy.loginfo(f"輸入裝置: {info['name']}")
        except Exception as e:
            rospy.logerr(f"PyAudio 初始化失敗: {e}")
            self.test_mode = True
    
    def _find_input_device(self, keywords=('pulse', 'bluetooth')):
        """自動搜尋輸入裝置"""
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            name = info['name'].lower()
            if info['maxInputChannels'] > 0 and any(kw in name for kw in keywords):
                return i
        return self.p.get_default_input_device_info()['index']
    
    def _log_test_mode_help(self):
        """輸出測試模式說明"""
        if not self.test_mode:
            return
        help_text = (
            "測試模式 - 鍵盤指令 (NSTC 114.12.19):\n"
            "  設備1 載具: 0=停止 1=開始/左右觀察 2=前進 3=左轉 4=右轉 5=左轉前進 6=右轉前進\n"
            "  設備2 攝影機 / 設備3 主機功能 / 設備4 執行裝置 [待硬體整合]"
        )
        rospy.loginfo(help_text)
    
    def system_status_callback(self, msg):
        """接收系統狀態回調"""
        if msg.data == "voice_command_executing" and not self.task_started:
            self.task_started = True
            rospy.loginfo("語音模組：已收到啟動訊號，任務標記為已開始")
        
        # 追踪避障功能狀態
        if msg.data == "fuzzy_control_active" and self.task_started:
            self.avoid_obstacle = True
            rospy.loginfo("語音模組：避障功能已啟用")
    
    def calculate_rms(self, data):
        """計算音頻塊的均方根值(RMS)，作為音量大小的指標"""
        try:
            # 將字節轉換為short整數數組
            shorts = np.frombuffer(data, dtype=np.int16)
            
            # 安全檢查：確保數據不為空
            if len(shorts) == 0:
                return 0.0
                
            # 計算平方
            squares = np.square(shorts.astype(np.float64))
            
            # 計算均值
            mean_squares = np.mean(squares)
            
            # 安全檢查：確保均值為正
            if mean_squares <= 0:
                return 0.0
                
            # 計算RMS
            rms = np.sqrt(mean_squares)
            return rms
        except Exception as e:
            rospy.logerr(f"計算RMS時出錯: {e}")
            return 0.0
    
    def record_audio_with_vad(self):
        """使用VAD算法進行智能錄音"""
        if self.test_mode:
            # 測試模式不實際錄音
            return self.wav_filename
            
        last_log_time = time.time()
        
        try:
            # 打開錄音流
            stream = self.p.open(format=self.FORMAT,
                                channels=self.CHANNELS,
                                rate=self.RATE,
                                input=True,
                                input_device_index=self.input_device_index,
                                frames_per_buffer=self.CHUNK)
            
            frames = []
            silent_chunks = 0
            voiced_chunks = 0
            recording = False
            buffer = []  # 用於保存可能的語音開始前的音頻(預緩衝)
            
            # 預緩衝區大小(約0.5秒)
            pre_buffer_size = int(0.5 * self.RATE / self.CHUNK)
            
            rospy.loginfo("開始監聽語音...")
            
            # 音量調試模式
            debug_volume = rospy.get_param('~debug_volume', False)
            if debug_volume:
                rospy.loginfo("音量調試模式已啟用，將顯示音量值以幫助調整閾值")
            
            start_time = time.time()
            max_time_reached = False
            
            while not self.stop_requested and not max_time_reached:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                current_rms = self.calculate_rms(data)

                # 每2秒輸出一次當前RMS音量
                if time.time() - last_log_time >= 2.0:
                    rospy.loginfo(f"即時音量 (RMS): {current_rms:.2f}")
                    last_log_time = time.time()
                
                # 如果啟用了音量調試模式，顯示當前音量值
                debug_volume = rospy.get_param('~debug_volume', False)
                if debug_volume:
                    # 只在音量超過一定值或每隔一段時間顯示，避免刷屏
                    if current_rms > 300 or int(time.time() * 2) % 10 == 0:
                        rospy.loginfo(f"當前音量: {current_rms:.2f}, 閾值: {self.SILENCE_THRESHOLD}")
                
                # 如果尚未開始錄製，將數據添加到緩衝區
                if not recording:
                    buffer.append(data)
                    # 保持緩衝區大小固定
                    if len(buffer) > pre_buffer_size:
                        buffer.pop(0)
                
                # 音量超過閾值，可能是語音
                if current_rms > self.SILENCE_THRESHOLD:
                    if not recording:
                        voiced_chunks += 1
                        if voiced_chunks >= self.SPEECH_START_THRESHOLD:
                            recording = True
                            # 添加預緩衝區的音頻到錄音中
                            frames.extend(buffer)
                            if self.show_prompt:
                                rospy.loginfo("檢測到語音，開始錄音...")
                            start_time = time.time()  # 重置開始時間
                    else:
                        # 已經在錄音中，重置靜音計數
                        silent_chunks = 0
                        frames.append(data)
                else:
                    # 音量低於閾值，可能是靜音
                    if recording:
                        silent_chunks += 1
                        frames.append(data)  # 仍然記錄靜音，以保持自然的語音結束
                        
                        if silent_chunks >= self.SPEECH_END_THRESHOLD:
                            if self.show_prompt:
                                rospy.loginfo("檢測到語音結束，停止錄音...")
                            break
                    else:
                        # 未在錄音但檢測到靜音，重置語音計數
                        voiced_chunks = 0
                
                # 檢查是否達到最長錄音時間
                if recording and (time.time() - start_time > self.MAX_RECORDING_SECONDS):
                    if self.show_prompt:
                        rospy.loginfo(f"達到最大錄音時間({self.MAX_RECORDING_SECONDS}秒)，停止錄音...")
                    max_time_reached = True
            
            # 關閉錄音流
            stream.stop_stream()
            stream.close()
            
            # 如果沒有檢測到有效語音，返回None
            if not recording or len(frames) < self.SPEECH_START_THRESHOLD:
                return None
            
            # 將錄音數據保存為WAV文件
            wf = wave.open(self.wav_filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            if self.show_prompt:
                rospy.loginfo(f"錄音已保存: {self.wav_filename}")
            
            return self.wav_filename
        except Exception as e:
            rospy.logerr(f"錄音過程中發生錯誤: {e}")
            return None
    
    def convert_wav_to_mp3(self, wav_filename):
        """使用 pydub 將 WAV 檔轉換成 mp3 格式，轉換結果存放於 BytesIO 中並保存一份到指定目錄"""
        try:
            sound = AudioSegment.from_wav(wav_filename)
            
            # 生成唯一的文件名，使用時間戳
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mp3_filename = os.path.join(self.recordings_dir, f"recording_{timestamp}.mp3")
            
            # 也返回內存中的MP3數據
            mp3_io = io.BytesIO()
            sound.export(mp3_io, format="mp3")
            return mp3_io.getvalue(), mp3_filename
        except Exception as e:
            rospy.logerr(f"轉換WAV到MP3時出錯: {e}")
            return None, None
    

    

    def get_audio_base64_data(self):
        """記錄音頻並轉換為base64編碼，同時保存一份到指定目錄"""
        if self.test_mode:
            # 在測試模式下返回空字符串
            return ""
                
        wav_file = self.record_audio_with_vad()
        if not wav_file:
            return None
            
        # ===== 加入噪音濾除 =====
        try:
            # 只有當成功導入必要的庫時才執行降噪
            import soundfile as sf
            import noisereduce as nr
            
            rospy.loginfo("開始進行噪聲濾除處理...")
            
            # 讀取剛錄製的 WAV
            y, sr = sf.read(wav_file)  
            
            # 取前 0.5 秒當作純噪聲樣本
            noise_clip = y[: int(0.5 * sr)]
            
            # 用 noisereduce 3.0.3 的 API
            # 正確格式: reduce_noise(y, sr, y_noise)
            reduced = nr.reduce_noise(y, sr, noise_clip)
            
            # 建立一個臨時文件名
            reduced_wav = os.path.join(self.recordings_dir, "reduced_recording.wav")
            
            # 存回降噪後的文件
            sf.write(reduced_wav, reduced, sr)
            rospy.loginfo(f"降噪完成，已存入: {reduced_wav}")
            
            # 使用降噪後的檔案來轉換 MP3
            wav_file = reduced_wav
        except Exception as e:
            rospy.logwarn(f"降噪失敗，保留原始錄音: {e}")
        # ===== 噪音濾除完成 =====

        # 接下來再轉 mp3、編 base64
        mp3_data, mp3_filename = self.convert_wav_to_mp3(wav_file)
        if not mp3_data:
            return None
                
        # 保存 WAV 和 MP3 到指定目錄
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存 WAV 檔案
            wav_copy_path = os.path.join(self.recordings_dir, f"recording_{timestamp}.wav")
            shutil.copy2(wav_file, wav_copy_path)
            rospy.loginfo(f"WAV 錄音已保存: {wav_copy_path}")
            
            # 保存 MP3 文件
            with open(mp3_filename, 'wb') as f:
                f.write(mp3_data)
            rospy.loginfo(f"MP3 錄音已保存: {mp3_filename}")
            
            # 刪除臨時文件
            if wav_file != self.wav_filename and os.path.exists(wav_file):
                os.remove(wav_file)
            if os.path.exists(self.wav_filename):
                os.remove(self.wav_filename)
        except Exception as e:
            rospy.logwarn(f"保存或刪除臨時文件時出錯: {e}")
                
        encoded_audio_data = base64.b64encode(mp3_data).decode('utf-8')
        return encoded_audio_data


    def send_audio_request(self, encoded_audio_data):
        """將 base64 編碼的音訊資料發送至語音服務，並回傳 JSON 格式的回應結果"""
        try:
            # 測試模式下模擬回應
            if self.test_mode:
                return {"response": {"inst_codes": []}}
                
            # 設置超時時間
            timeout = 45  # 5秒超時
            
            # 組建 URL（標準 HTTPS port 不需指定）
            if AMRR_VOICE_SERVICE_PORT and AMRR_VOICE_SERVICE_PORT != 443:
                api_url = f'https://{AMRR_VOICE_SERVICE_HOST}:{AMRR_VOICE_SERVICE_PORT}/IAAgent/transcribe_translate'
            else:
                api_url = f'https://{AMRR_VOICE_SERVICE_HOST}/IAAgent/transcribe_translate'
            
            response = requests.post(
                api_url,
                json = {
                    "serviceid": AMRR_VOICE_SERVICE_ID,
                    "func": "amrr-001",
                    "conn": "ntnu-12345",
                    "invoice": encoded_audio_data  
                },
                timeout=timeout
            )
            
            if response.status_code != 200:
                rospy.logwarn(f"語音服務返回非200狀態碼: {response.status_code}")
                return {"response": {}}
                
            return response.json()
        except requests.exceptions.RequestException as e:
            rospy.logerr(f"發送語音請求時出錯: {e}")
            return {"response": {}}
        except json.JSONDecodeError as e:
            rospy.logerr(f"解析JSON響應時出錯: {e}")
            return {"response": {}}
        except Exception as e:
            rospy.logerr(f"發送語音請求時出錯: {e}")
            return {"response": {}}
    
    def process_audio_response(self, jresult):
        """處理語音服務回應"""
        responses = jresult.get('response', {})
        transcription = responses.get('transcription', '')
        inst_codes = responses.get('inst_codes', [])
        
        rospy.loginfo(f"辨識文字: {transcription}")
        
        if not inst_codes:
            return
        
        # 冷卻時間檢查
        current_time = time.time()
        if current_time - self.last_cmd_time < self.cmd_cooldown:
            rospy.loginfo_throttle(1.0, "指令冷卻中")
            return
        
        # 驗證並處理指令
        for code in inst_codes:
            code_str = str(code).zfill(5)
            
            if not self._is_valid_code(code_str):
                rospy.loginfo(f"忽略無效編碼: {code_str}")
                continue
            
            self.last_cmd_time = current_time
            self._execute_voice_code(code_str)
    
    def _is_valid_code(self, code):
        """驗證指令編碼是否有效 (支援 4 類設備)"""
        if len(code) != 5:
            return False
        
        device_code = code[0]   # 設備碼
        action_code = code[1]   # 動作碼
        
        # 設備 1：載具控制
        if device_code == '1':
            return action_code in VEHICLE_ACTIONS
        
        # 設備 2：攝影機觀察 (含 21000 左右觀察由載具執行)
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
    
    def _execute_voice_code(self, code):
        """執行語音指令 (支援 4 類設備)"""
        device_code = code[0]
        action_code = code[1]
        param_code = code[2:5]
        
        # === 設備 1：載具控制 ===
        if device_code == '1':
            action_name = VEHICLE_ACTIONS.get(action_code, '未知動作')
            
            if action_code == '0':  # 暫停待命
                self.avoid_obstacle = False
                rospy.loginfo("載具：暫停待命")
            elif action_code == '1':  # 開始巡邏 / 前進
                if code == '11000':  # 開始巡邏：先左右觀察，再前進避障
                    self.task_started = True
                    self.avoid_obstacle = True
                    rospy.loginfo("載具：開始巡邏 (左右觀察90° → 前進避障)")
                else:  # 一般前進指令 (含距離參數)
                    if self.task_started:
                        self.avoid_obstacle = True
                    rospy.loginfo(f"載具：前進 (參數: {param_code})")
            elif action_code == '2':  # 繼續巡邏
                rospy.loginfo("載具：繼續巡邏")
            elif action_code == '3':  # 左轉
                rospy.loginfo(f"載具：左轉 (參數: {param_code})")
            elif action_code == '4':  # 右轉
                rospy.loginfo(f"載具：右轉 (參數: {param_code})")
            elif action_code in ('5', '6'):  # 轉彎後前進
                if self.task_started:
                    self.avoid_obstacle = True
                rospy.loginfo(f"載具：{action_name} (參數: {param_code})")
            elif action_code == '7':  # 播放影片
                video_name = VIDEO_CODES.get(param_code[1:3], '未知影片')
                rospy.loginfo(f"載具：播放影片 [{video_name}]")
        
        # === 設備 2：攝影機 (含 21000 左右觀察) ===
        elif device_code == '2':
            if code in VEHICLE_SPECIAL_CODES:  # 21000 左右觀察 (載具執行)
                self.task_started = True
                self.avoid_obstacle = False
                rospy.loginfo("載具：左右觀察 (左右轉90°)")
            else:
                action_name = CAMERA_ACTIONS.get(action_code, '未知動作')
                rospy.loginfo(f"攝影機：{action_name} (參數: {param_code}) [待硬體整合]")
        
        # === 設備 3：主機功能 ===
        elif device_code == '3':
            action_name = HOST_ACTIONS.get(action_code, '未知動作')
            rospy.loginfo(f"主機功能：{action_name} (參數: {param_code}) [待硬體整合]")
        
        # === 設備 4：執行裝置 ===
        elif device_code == '4':
            action_name = ACTUATOR_ACTIONS.get(action_code, '未知動作')
            if action_code == '5':  # 拿給我
                item_name = ITEM_CODES.get(param_code[1:3], '未知物品')
                rospy.loginfo(f"執行裝置：拿給我 [{item_name}] [待硬體整合]")
            else:
                rospy.loginfo(f"執行裝置：{action_name} [待硬體整合]")
        
        # 發布指令到 ROS Topic
        self.voice_pub.publish(code)
        rospy.loginfo(f"發布編碼: {code}")
        
        # 通知系統狀態 (停止指令除外)
        if code != "10000":
            self.system_status_pub.publish("voice_command_executing")
    
    def voice_command_callback(self, msg):
        """接收語音命令回調"""
        code = msg.data.strip()
        
        if len(code) < 5:
            rospy.logwarn(f"編碼長度不足: {code}")
            return
        
        if not self._is_valid_code(code):
            rospy.loginfo_throttle(1.0, f"忽略無效編碼: {code}")
            return
        
        self._execute_voice_code(code)
    
    def simulate_voice_command(self, key):
        """模擬語音指令 - 測試模式使用"""
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
        
        if key in code_map:
            code = code_map[key]
            
            # 檢查冷卻時間
            current_time = time.time()
            if current_time - self.last_cmd_time < self.cmd_cooldown:
                rospy.loginfo(f"指令冷卻中，跳過指令: {code}")
                return
            
            # 更新指令時間
            self.last_cmd_time = current_time
            
            # 輸出接收到的指令
            rospy.loginfo(f"接收到模擬指令編碼: {code}")
            
            # 發布指令
            self.voice_pub.publish(code)
            rospy.loginfo(f"已發布模擬語音編碼：{code}")
            
            # 特殊處理「開始任務」指令
            if code == "21000":
                rospy.loginfo("收到「開始任務」指令")
                self.task_started = True
                self.avoid_obstacle = False
            
            # 如果是移動相關指令，標記避障功能啟用
            if self.task_started and (code.startswith("11") or code.startswith("15") or code.startswith("16")):
                self.avoid_obstacle = True
                rospy.loginfo("收到移動指令，標記為需要啟用避障功能")
            
            # 如果是停止指令，標記避障功能關閉
            if code == "10000":
                self.avoid_obstacle = False
                rospy.loginfo("收到停止指令，標記為關閉避障功能")
            else:
                # 通知系統狀態變更
                self.system_status_pub.publish("voice_command_executing")
        else:
            rospy.loginfo(f"無效的模擬指令：{key}，有效值為：0-9")
    
    def test_mode_input(self):
        """在測試模式下監聽鍵盤輸入"""
        while not rospy.is_shutdown() and not self.stop_requested:
            try:
                if sys.stdin.isatty():  # 確保有可用的標準輸入
                    rospy.loginfo("請輸入一個數字(0-9)來模擬語音指令:")
                    key = input()
                    self.simulate_voice_command(key)
                else:
                    # 如果沒有標準輸入，使用預定義的按鍵序列
                    for key in ["1", "2", "3", "4", "5", "6", "0"]:  # 開始任務、前進、左轉、右轉、左轉前進、右轉前進、停止
                        if self.stop_requested:
                            break
                        rospy.loginfo(f"自動模擬指令：{key}")
                        self.simulate_voice_command(key)
                        rospy.sleep(10)  # 10秒間隔
            except KeyboardInterrupt:
                rospy.loginfo("測試模式被用戶中斷")
                self.stop_requested = True
            except Exception as e:
                rospy.logerr(f"測試模式輸入時出錯: {e}")
                rospy.sleep(5)  # 錯誤後等待5秒再重試
    
    def listen_continuously(self):
        """持續監聽語音指令的線程"""
        if self.test_mode:
            self.test_mode_input()
            return

        while not rospy.is_shutdown() and not self.stop_requested:
            if not self.is_processing:
                self.is_processing = True
                try:
                    # 設置不顯示提示信息，以避免控制台信息過多
                    self.show_prompt = False
                    
                    # 獲取語音數據
                    encoded_audio_data = self.get_audio_base64_data()
                    if encoded_audio_data:
                        # 發送語音請求
                        jresult = self.send_audio_request(encoded_audio_data)
                        
                        # 處理回應
                        self.process_audio_response(jresult)
                    
                except Exception as e:
                    rospy.logerr(f"語音處理過程中出錯: {e}")
                finally:
                    self.is_processing = False
                    # 短暫休息以減少CPU使用率
                    rospy.sleep(0.1)
            else:
                # 如果正在處理，短暫休息
                rospy.sleep(0.1)
    
    def run(self):
        """主運行函數"""
        rospy.loginfo("語音指令節點運行中...")
        
        # 創建一個線程用於持續監聽語音指令
        listen_thread = threading.Thread(target=self.listen_continuously)
        listen_thread.daemon = True
        listen_thread.start()
        
        # 主線程等待 ROS 關閉
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("使用者中斷")
        finally:
            self.stop_requested = True
            if not self.test_mode and hasattr(self, 'p'):
                try:
                    self.p.terminate()
                except Exception as e:
                    rospy.logwarn(f"PyAudio 終止失敗: {e}")
            rospy.loginfo("語音節點已關閉")

def main():
    try:
        node = VoiceCommandNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("語音指令節點被中斷")
    except Exception as e:
        rospy.logerr(f"語音指令節點發生錯誤: {e}")

if __name__ == "__main__":
    main()
