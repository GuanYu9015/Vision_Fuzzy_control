# [cite_start]機器人指令轉譯表 (114.12.19 版) 

> [cite_start]**單位**：NSTC 國家科學及技術委員會 [cite: 2, 6, 11, 14]

---

## 一、 指令轉譯對照表

### [cite_start]1. 載具控制 (設備碼: 1) [cite: 3, 20]
| 動作 (Action) | 動作碼 (Action Code) | 參數 (Argument) | 參數碼 (Argument Code) |
| :--- | :---: | :--- | :---: |
| [cite_start]開始巡邏 [cite: 3] | [cite_start]1 [cite: 3] | [cite_start]無 [cite: 3] | [cite_start]0 0 0 [cite: 3] |
| [cite_start]繼續巡邏 [cite: 3] | [cite_start]2 [cite: 3] | [cite_start]無 [cite: 3] | [cite_start]0 0 0 [cite: 3] |
| [cite_start]前進到 / 引導到 / 去★ [cite: 3] | [cite_start]1 [cite: 3] | [cite_start]@@公尺(距離) / 到★(地點) [cite: 3] | [cite_start]@ @ 1 / 2 ★ [cite: 3] |
| [cite_start]左轉 [cite: 3] | [cite_start]3 [cite: 3] | [cite_start]@@度(角度) [cite: 3] | [cite_start]@ @ 1 [cite: 3] |
| [cite_start]右轉 [cite: 3] | [cite_start]4 [cite: 3] | [cite_start]無 [cite: 3] | [cite_start]0 0 0 [cite: 3] |
| [cite_start]左轉(90度)往前 [cite: 3] | [cite_start]5 [cite: 3] | [cite_start]到★(地點) / 無 [cite: 3] | [cite_start]★ 2 0 / 0 0 [cite: 3] |
| [cite_start]右轉(90度)往前 [cite: 3] | [cite_start]6 [cite: 3] | [cite_start]到★(地點) / 無 [cite: 3] | [cite_start]★ 2 0 / 0 0 [cite: 3] |
| [cite_start]播放影片 [cite: 3] | [cite_start]7 [cite: 3] | (影片代碼) [cite_start][cite: 3] | [cite_start]1 [cite: 3] |
| [cite_start]暫停、待命 [cite: 3] | [cite_start]0 [cite: 3] | [cite_start]無 [cite: 3] | [cite_start]0 0 0 [cite: 3] |

### [cite_start]2. 攝影機觀察 (設備碼: 2) [cite: 7, 24]
| 動作 (Action) | 動作碼 (Action Code) | 參數 (Argument) | 參數碼 (Argument Code) |
| :--- | :---: | :--- | :---: |
| [cite_start]觀察環境、左右觀察 [cite: 7] | [cite_start]1 [cite: 7] | [cite_start]無 (左右各90度) [cite: 7] | [cite_start]0 0 0 [cite: 7] |
| [cite_start]上下觀察 [cite: 7] | [cite_start]2 [cite: 7] | [cite_start]無 (上下各45度) [cite: 7] | [cite_start]0 0 0 [cite: 7] |
| [cite_start]鏡頭往左 / 右 [cite: 7] | [cite_start]3 / 4 [cite: 7] | [cite_start]@@度 (角度) [cite: 7] | [cite_start]@ 1 @ [cite: 7] |
| [cite_start]鏡頭往上 [cite: 7] | [cite_start]5 [cite: 7] | [cite_start]到★ (目標物) [cite: 7] | [cite_start]2 ★ [cite: 7] |
| [cite_start]鏡頭往下 [cite: 7] | [cite_start]6 [cite: 7] | [cite_start]無 [cite: 7] | [cite_start]0 0 0 [cite: 7] |

### [cite_start]3. 主機功能與辨識 (設備碼: 3) [cite: 8, 25]
| 動作 (Action) | 動作碼 (Action Code) | 參數 (Argument) | 參數碼 (Argument Code) |
| :--- | :---: | :--- | :---: |
| [cite_start]回報 (狀況) [cite: 8] | [cite_start]1 [cite: 8] | [cite_start]人員 (01) / 環境 (02) / 無 (00) [cite: 8] | [cite_start]0 1 0 / 0 2 0 / 0 0 0 [cite: 8] |
| [cite_start]識別、搜尋 [cite: 8] | [cite_start]2 [cite: 8] | [cite_start]★ (目標物) [cite: 8] | [cite_start]2 ★ [cite: 8] |
| [cite_start]測量 [cite: 8] | [cite_start]3 [cite: 8] | [cite_start]走道 (長度) [cite: 8] | [cite_start]0 1 0 [cite: 8] |
| [cite_start]通知、預約 [cite: 8] | [cite_start]4 [cite: 8] | (科室 / 單位) [cite_start][cite: 8] | [cite_start]1 ★ [cite: 8] |
| [cite_start]紀錄 (拍照 / 錄影) [cite: 8] | [cite_start]5 [cite: 8] | [cite_start]拍照 (01) / 錄影 (02) [cite: 8] | [cite_start]0 1 0 / 0 2 0 [cite: 8] |

### [cite_start]4. 執行裝置 (設備碼: 4) [cite: 12, 26]
| 動作 (Action) | 動作碼 (Action Code) | 參數 (Argument) | 參數碼 (Argument Code) |
| :--- | :---: | :--- | :---: |
| [cite_start]開門 [cite: 12] | [cite_start]1 [cite: 12] | [cite_start]無 [cite: 12] | [cite_start]0 0 0 [cite: 12] |
| [cite_start]關門 [cite: 12] | [cite_start]2 [cite: 12] | [cite_start]無 [cite: 12] | [cite_start]0 0 0 [cite: 12] |
| [cite_start]解鎖 [cite: 12] | [cite_start]3 [cite: 12] | [cite_start]無 [cite: 12] | [cite_start]0 0 0 [cite: 12] |
| [cite_start]上鎖 [cite: 12] | [cite_start]4 [cite: 12] | [cite_start]無 [cite: 12] | [cite_start]0 0 0 [cite: 12] |
| [cite_start]拿給我 [cite: 12] | [cite_start]5 [cite: 12] | [cite_start]▲ (目標物) [cite: 12] | [cite_start]1 A [cite: 12] |

---

## 二、 參數代碼表 (Argument Codes)

### [cite_start]1. 地點 / 目標物 (Code: ★) [cite: 21]
| 地點名稱 | 代碼 (Code) | 地點名稱 | 代碼 (Code) |
| :--- | :---: | :--- | :---: |
| [cite_start]傷患 [cite: 21] | [cite_start]01 [cite: 21] | [cite_start]斜坡 [cite: 21] | [cite_start]02 [cite: 21] |
| [cite_start]門口 [cite: 21] | [cite_start]03 [cite: 21] | [cite_start]轉角處 [cite: 21] | [cite_start]04 [cite: 21] |
| [cite_start]廁所 [cite: 21] | [cite_start]05 [cite: 21] | [cite_start]電梯 [cite: 21] | [cite_start]06 [cite: 21] |
| [cite_start]樓梯 [cite: 21] | [cite_start]07 [cite: 21] | [cite_start]病房 [cite: 21] | [cite_start]08 [cite: 21] |
| [cite_start]護理站 [cite: 21] | [cite_start]09 [cite: 21] | [cite_start]急診室 [cite: 21] | [cite_start]10 [cite: 21] |
| [cite_start]X光室 [cite: 21] | [cite_start]11 [cite: 21] | | |

### [cite_start]2. 影片內容 (Code: ★) [cite: 22]
| 影片名稱 | 代碼 (Code) |
| :--- | :---: |
| [cite_start]就診流程 [cite: 22] | [cite_start]01 [cite: 22] |
| [cite_start]火災疏散 [cite: 22] | [cite_start]02 [cite: 22] |

### [cite_start]3. 物品名稱 (Code: ▲) [cite: 23]
| 物品名稱 | 代碼 (Code) | 物品名稱 | 代碼 (Code) |
| :--- | :---: | :--- | :---: |
| [cite_start]水 [cite: 23] | [cite_start]01 [cite: 23] | [cite_start]咖啡 [cite: 23] | [cite_start]02 [cite: 23] |
| [cite_start]茶 [cite: 23] | [cite_start]03 [cite: 23] | [cite_start]汽水 [cite: 23] | [cite_start]04 [cite: 23] |
| [cite_start]碗 [cite: 23] | [cite_start]05 [cite: 23] | [cite_start]杯子 [cite: 23] | [cite_start]06 [cite: 23] |
| [cite_start]巧克力 [cite: 23] | [cite_start]07 [cite: 23] | [cite_start]蘋果 [cite: 23] | [cite_start]08 [cite: 23] |
| [cite_start]優格 [cite: 23] | [cite_start]09 [cite: 23] | | |