import cv2
from pathlib import Path

# ==========================================
# 1. 參數設定區 (每次跑新影片只需改這裡)
# ==========================================
VIDEO_FILENAME = r"IMG_6836.mp4"
OUTPUT_SUFFIX = "02"  # 填寫後會自動建立 chen_xxx 資料夾

# ==========================================
# 2. 系統路徑設定
# ==========================================
BASE_SRC_DIR = Path(r"E:\2025_09_splash_dataset_unlabel")
BASE_DST_DIR = Path(r"C:\Users\user\PycharmProjects\meta_expand_test")


def main():
    video_path = BASE_SRC_DIR / VIDEO_FILENAME
    out_dir = BASE_DST_DIR / f"chen_{OUTPUT_SUFFIX}"

    if not video_path.exists():
        print(f"錯誤：找不到影片 {video_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_name = 'Manual Frame Extractor (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # 建立進度條，但不綁定回調函數，改在迴圈內主動偵測
    cv2.createTrackbar('Frame', window_name, 0, total_frames - 1, lambda x: None)

    current_frame_idx = 0
    last_frame_idx = -1  # 用來記錄上一幀，避免重複讀取浪費資源
    saved_count = 0
    ret = False
    frame = None

    print("=========================================")
    print("操作說明：")
    print("滑鼠：拖曳上方進度條可快速即時跳轉")
    print("按鍵 [A]：回退一幀")
    print("按鍵 [D]：前進一幀")
    print("按鍵 [S]：儲存當前畫面")
    print("按鍵 [Q] 或 [ESC]：離開程式")
    print("=========================================")

    while True:
        # 同步進度條與當前幀數
        trackbar_pos = cv2.getTrackbarPos('Frame', window_name)
        if trackbar_pos != current_frame_idx:
            current_frame_idx = trackbar_pos

        # 只有當「目標幀數」改變時，才重新從影片中讀取畫面
        if current_frame_idx != last_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()

            if ret:
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Frame: {current_frame_idx}/{total_frames}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(window_name, display_frame)

            last_frame_idx = current_frame_idx

        # 等待 30 毫秒，這樣既能捕捉鍵盤，又能讓滑鼠拖曳有反應
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('d'):
            current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
            cv2.setTrackbarPos('Frame', window_name, current_frame_idx)
        elif key == ord('a'):
            current_frame_idx = max(current_frame_idx - 1, 0)
            cv2.setTrackbarPos('Frame', window_name, current_frame_idx)
        elif key == ord('s'):
            if ret and frame is not None:
                out_name = f"{Path(VIDEO_FILENAME).stem}_{current_frame_idx:06d}.png"
                out_path = out_dir / out_name
                cv2.imwrite(str(out_path), frame)
                saved_count += 1
                print(f"✅ 已儲存: {out_name} (目前共挑選 {saved_count} 張)")

                # 存完自動往前一幀
                current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
                cv2.setTrackbarPos('Frame', window_name, current_frame_idx)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n標記結束！本次共手動挑選了 {saved_count} 張圖片，存於 {out_dir}")


if __name__ == "__main__":
    main()