import os
import json
import numpy as np
import cv2

# 設定目標目錄
TARGET_DIR = r"/mask_prompt"


def convert_json_to_mask(target_dir):
    if not os.path.exists(target_dir):
        print(f"錯誤: 找不到目錄 {target_dir}")
        return

    # 取得所有 json 檔案
    json_files = [f for f in os.listdir(target_dir) if f.endswith('.json')]

    if not json_files:
        print("目錄中沒有找到 JSON 檔案。")
        return

    print(f"找到 {len(json_files)} 個 JSON 檔案，開始轉換...")

    for json_file in json_files:
        json_path = os.path.join(target_dir, json_file)
        base_name = os.path.splitext(json_file)[0]

        # 依照你的命名習慣加上 _mask
        mask_filename = f"{base_name}_mask.png"
        mask_path = os.path.join(target_dir, mask_filename)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 讀取 JSON 內的影像長寬
        img_height = data.get('imageHeight')
        img_width = data.get('imageWidth')

        if img_height is None or img_width is None:
            print(f"警告: {json_file} 缺少影像長寬資訊，已跳過。")
            continue

        # 建立全黑遮罩 (背景 0 對應 'unlabel')
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # 遍歷所有的標註形狀
        for shape in data.get('shapes', []):
            label = shape.get('label', '')

            # 只有一個 ripple 類別，符合時填入數值 1
            if label.lower() == 'ripple':
                points = np.array(shape.get('points'), dtype=np.int32)
                # cv2.fillPoly 需傳入 list of polygons
                cv2.fillPoly(mask, [points], color=1)

        # 儲存轉換後的 Mask
        cv2.imwrite(mask_path, mask)
        print(f"轉換完成: {mask_filename}")

    print("所有檔案轉換完畢。")


if __name__ == "__main__":
    convert_json_to_mask(TARGET_DIR)