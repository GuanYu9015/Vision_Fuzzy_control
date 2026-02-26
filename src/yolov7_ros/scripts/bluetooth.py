# -*- coding: utf-8 -*-
import pyaudio
import wave

# 參數設定
DEVICE_INDEX = 28        # 由剛才列出來的 [25] pulse
RATE = 44100             # 取樣頻率
CHANNELS = 1             # 單聲道
FORMAT = pyaudio.paInt16 # 16-bit PCM
FRAMES_PER_BUFFER = 1024 # 緩衝大小
RECORD_SECONDS = 5       # 錄 5 秒
OUTPUT_FILENAME = "test_record.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=FRAMES_PER_BUFFER)

print(f"開始錄音 {RECORD_SECONDS} 秒…(按 Ctrl+C 可中斷)")

frames = []
for _ in range(0, int(RATE / FRAMES_PER_BUFFER * RECORD_SECONDS)):
    data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
    frames.append(data)

print("錄音結束，寫入檔案中…")
stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"已儲存：{OUTPUT_FILENAME}")
