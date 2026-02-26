import cv2
import numpy as np

def create_chessboard(pattern_size=(9, 6), square_size_mm=25, dpi=300):
    """
    生成相機校正用的棋盤格圖片
    :param pattern_size: (columns, rows) 內角點數量，例如 (9, 6)
    :param square_size_mm: 每個格子的邊長 (mm)，預設 25mm
    :param dpi: 列印解析度，預設 300 DPI (高品質列印)
    """
    
    # 1. 計算像素參數
    # 1 inch = 25.4 mm
    px_per_mm = dpi / 25.4
    square_px = int(square_size_mm * px_per_mm)
    
    # 內角點 (9, 6) 意味著格子數要是 (10, 7)
    width_squares = pattern_size[0] + 1
    height_squares = pattern_size[1] + 1
    
    # 2. 設定畫布大小 (外加 2 格寬度的邊距 Margin)
    margin_squares = 2
    img_width = (width_squares + margin_squares) * square_px
    img_height = (height_squares + margin_squares) * square_px
    
    # 建立白色背景 (單通道灰階圖)
    image = np.ones((img_height, img_width), dtype=np.uint8) * 255
    
    # 3. 繪製黑色方塊
    # 起始偏移量 (margin / 2)
    start_x = int(margin_squares / 2 * square_px)
    start_y = int(margin_squares / 2 * square_px)
    
    for r in range(height_squares):
        for c in range(width_squares):
            # 偶數行偶數列、奇數行奇數列 為黑色 (交錯圖案)
            if (r + c) % 2 == 1:
                pt1 = (start_x + c * square_px, start_y + r * square_px)
                pt2 = (pt1[0] + square_px, pt1[1] + square_px)
                cv2.rectangle(image, pt1, pt2, color=0, thickness=-1) # -1 代表填滿
    
    return image

# --- 主程式 ---
if __name__ == "__main__":
    # 設定參數：內角點 9x6，每格 2.5 公分 (25mm)
    # 這是最通用的校正板規格
    cols = 9
    rows = 6
    sq_size = 25 # mm
    
    print(f"正在生成 {cols}x{rows} (內角點) 的棋盤格...")
    print(f"格子大小: {sq_size}mm (請列印時選擇 '原始大小/100%' 勿縮放)")

    board_img = create_chessboard(pattern_size=(cols, rows), square_size_mm=sq_size)
    
    # 儲存檔案
    filename = f"chessboard_{cols}x{rows}_{sq_size}mm.png"
    cv2.imwrite(filename, board_img)
    print(f"成功儲存: {filename}")
    
    # 顯示預覽 (縮小顯示以免螢幕塞不下)
    preview = cv2.resize(board_img, (0,0), fx=0.2, fy=0.2)
    cv2.imshow("Preview (Press any key to exit)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()