#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
--2026/03/13--純 Attention 架構
"""
import os
import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

def fix_seeds(seed: int = 3407) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

fix_seeds(3402)


# # 資料載入
# 移除舊有的線性代數協方差運算（計算 P, N, r_basis 等），並將 K-means 的對象從「像素級別（Pixel-level）的 64 維機率分佈」，升級為「影像級別（Image-level）的 256 維水花特徵表徵」。
# <br>
# 系統將讀取 .pt 檔（$256 \times 200 \times 200$），利用對應的標註遮罩（Mask）過濾出水花區域，然後將該區域的 256 維特徵進行平均，壓縮成一組代表該影像水花的 $1 \times 256$ 向量，再以這組向量進行 K-means 分群。

# In[2]:


# ================= 1. 資料準備與特徵池化 (GAP) =================
print("--- 步驟 1：載入 .pt 特徵並進行水花區域 GAP (Global Average Pooling) ---")

root_path = r"C:\Users\user\PycharmProjects\organized_ripple_4fold"
fields = ['barrel', 'sea', 'LNG', 'seabass_hmh', 'noon_jsj']

all_features_list = []
original_image_paths = [] # 紀錄來源，方便後續還原為 RGB 影像交給 Gemini 標註

for field in fields:
    feat_dir = Path(root_path) / field / 'features_256'
    lbl_dir = Path(root_path) / field / 'labels_detectron2'

    if not feat_dir.exists() or not lbl_dir.exists():
        print(f"警告：找不到場域 {field} 的特徵或標註資料夾，已跳過。")
        continue

    pt_files = sorted(list(feat_dir.glob('*.pt')))

    for pt_path in pt_files:
        lbl_path = lbl_dir / f"{pt_path.stem}.png"

        if not lbl_path.exists():
            continue

        # 讀取特徵圖 (256, 200, 200)
        feature_map = torch.load(pt_path, map_location='cpu')

        # 讀取 Ground Truth 標籤 (H, W)，假設 1 為水花，0 為背景
        label = torch.tensor(np.array(Image.open(lbl_path)))

        # 將標籤 Resize 以對齊特徵圖的空間大小 (200, 200)
        label_resized = F.interpolate(
            label.unsqueeze(0).unsqueeze(0).float(),
            size=feature_map.shape[1:],
            mode='nearest'
        ).squeeze().long()

        # 提取水花區域 Mask
        ripple_mask = (label_resized == 1)

        # [新增] 這就是您的 gt.sum().item()，代表水花像素總面積 (Intensity)
        splash_pixel_count = ripple_mask.sum().item()

        if splash_pixel_count == 0:
            continue # 此影像無明顯水花，跳過

        # GAP 計算 (256, )
        ripple_features = feature_map[:, ripple_mask]
        gap_feature = ripple_features.mean(dim=1)

        all_features_list.append(gap_feature.numpy())

        # [修改] 將 intensity 一併記錄到原始資訊字典中
        original_image_paths.append({
            'patch_id': len(original_image_paths), # 賦予輕量級的 ID
            'field': field,
            'pt_path': str(pt_path),
            'img_name': f"{pt_path.stem}.jpg",
            'intensity': splash_pixel_count  # 保留供 Graph RAG 排序使用
        })

all_features_np = np.vstack(all_features_list)
print(f"--- 步驟 1：提取完成。成功處理 {len(all_features_list)} 張有效水花影像。 特徵矩陣 Shape: {all_features_np.shape} ---")


# In[3]:


# ================= 2. 資料前處理 =================
print("--- 步驟 2: 特徵正規化 (L2 Norm) ---")
# 確保特徵投影在單位球面上，使歐式距離等價於餘弦相似度排序
all_features_norm = normalize(all_features_np, norm='l2', axis=1)


# # K-means

# In[4]:


# ================= 3. K 值自動化 (Silhouette) =================
print("--- 步驟 3: 自動尋找最佳 K 值 ---")
range_n_clusters = [6, 8, 10, 12, 16] # 考量 256 維與影像級別，稍微下調 K 值搜尋範圍
best_k = 8
best_score = -1
scores = []

# 若樣本過多則隨機抽樣計算輪廓係數以加速
sample_size = min(10000, all_features_norm.shape[0])
indices = np.random.choice(all_features_norm.shape[0], sample_size, replace=False)
sample_data = all_features_norm[indices]

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=3407, n_init='auto')
    cluster_labels = clusterer.fit_predict(sample_data)

    silhouette_avg = silhouette_score(sample_data, cluster_labels)
    scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, Silhouette Score : {silhouette_avg:.4f}")

    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_k = n_clusters

print(f"===> 最佳 K 值為: {best_k} (Score: {best_score:.4f})")


# In[5]:


# ================= 4. 執行最終 K-Means 與計算代表點 (Medoid) =================
print(f"--- 步驟 4: 使用 K={best_k} 對全量數據分群並尋找代表點 ---")
kmeans = KMeans(n_clusters=best_k, random_state=3407, n_init=10)
labels = kmeans.fit_predict(all_features_norm)

visual_bases_medoids = []
priors = []
best_representative_images = [] # 儲存將提供給 Gemini 標註的原始影像資訊

total_samples = all_features_norm.shape[0]

for i in range(best_k):
    indices = np.where(labels == i)[0]
    cluster_points = all_features_norm[indices]

    # 計算 Prior (該群佔比)
    prob = len(indices) / total_samples
    priors.append(prob)

    # 尋找距離質心最近的真實資料點 (Medoid)
    centroid = kmeans.cluster_centers_[i].reshape(1, -1)
    dists = euclidean_distances(cluster_points, centroid)
    min_dist_idx = np.argmin(dists)

    medoid_vector = cluster_points[min_dist_idx]
    visual_bases_medoids.append(medoid_vector)

    # 取得原始影像資訊，此影像將做為該群的最佳代表，交由 Gemini 進行標註
    original_idx = indices[min_dist_idx]
    best_image_info = original_image_paths[original_idx]
    best_representative_images.append(best_image_info)

    print(f"Cluster {i:02d}: Count={len(indices)}, Prior={prob:.4f}, 代表影像={best_image_info['img_name']}")

visual_bases_tensor = torch.tensor(np.array(visual_bases_medoids)).float()
priors_tensor = torch.tensor(np.array(priors)).float()


# In[6]:


# ================= 5. 儲存結果 =================
print("--- 步驟 5: 儲存分群結果與代表點資訊 ---")
output_dir = "prompt_256D"
os.makedirs(output_dir, exist_ok=True)

torch.save(visual_bases_tensor, os.path.join(output_dir, f"visual_bases_K{best_k}_medoid.pth"))
torch.save(priors_tensor, os.path.join(output_dir, f"priors_K{best_k}.pth"))
np.save(os.path.join(output_dir, "kmeans_labels.base_npy"), labels)

# 將代表點的影像路徑儲存為 List，方便下一階段自動載入圖片並呼叫 Gemini API
import json
with open(os.path.join(output_dir, "best_representative_images.json"), 'w') as f:
    json.dump(best_representative_images, f, indent=4)

print("流程執行完畢，所有結果已儲存至:", output_dir)


# # Visualize

# In[7]:


import os
import json
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path

# ================= 設定區域 =================
OUTPUT_DIR = "vlm_knowledge_construction_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 64
ANCHOR_SIZE = 256
GLOBAL_RESIZE = 512

# --- 1. 專家知識注入 (Expert Knowledge Injection) ---
FIELD_INTENSITY_RANGE = {
    "A": (1, 3), "B": (7, 10), "C": (3, 6), "D": (2, 5), "E": (1, 3)
}

FIELD_METADATA_MAPPING = {
    "barrel": {
        "field_id": "A",
        "intensity_baseline": 3,
        "environment": "outdoor_small_plastic_tank",
        "fish_species": ["black_seabream", "seabass_fry"], # 拆分為陣列
        "perspective": "near_overhead"
    },
    "sea": {
        "field_id": "B",
        "intensity_baseline": 10,
        "environment": "offshore_cage",
        "fish_species": ["cobia", "pompano"], # 拆分為陣列
        "perspective": "medium_oblique"
    },
    "LNG": {
        "field_id": "C",
        "intensity_baseline": 5,
        "environment": "indoor_concrete_pond",
        "fish_species": ["hybrid_rock_bream"], # 單一魚種也用陣列包裝，維持資料結構一致性
        "perspective": "medium_oblique"
    },
    "seabass_hmh": {
        "field_id": "D",
        "intensity_baseline": 4,
        "environment": "onshore_concrete_pond",
        "fish_species": ["seabass"],
        "perspective": "far_distant"
    },
    "noon_jsj": {
        "field_id": "E",
        "intensity_baseline": 1,
        "environment": "onshore_concrete_pond",
        "fish_species": ["fourfinger_threadfin"],
        "perspective": "medium_oblique"
    }
}

def calculate_intensity_stats(source_counts, total_count):
    weighted_sum = 0.0
    min_scores, max_scores = [], []
    for field_id, count in source_counts.items():
        weight = count / total_count
        baseline = next((m['intensity_baseline'] for m in FIELD_METADATA_MAPPING.values() if m['field_id'] == field_id), 0)
        weighted_sum += baseline * weight
        if weight > 0.1 and field_id in FIELD_INTENSITY_RANGE:
            min_scores.append(FIELD_INTENSITY_RANGE[field_id][0])
            max_scores.append(FIELD_INTENSITY_RANGE[field_id][1])
    return int(round(weighted_sum)), min(min_scores) if min_scores else 0, max(max_scores) if max_scores else 0

# --- 2. 影像處理輔助函式 (基於 GT Mask 尋找質心並支援多重副檔名) ---
def read_and_process_image(field_name, img_name, root_path=r"C:\Users\user\PycharmProjects\organized_ripple_4fold"):
    # 剝除舊有錯誤的副檔名，取得純檔名 (例如: barrel0924_1148_1_421)
    base_name = Path(img_name).stem

    # 動態路徑尋找 (先找 jpg，沒有再找 png)
    img_dir = Path(root_path) / field_name / 'images'
    img_path_jpg = img_dir / f"{base_name}.jpg"
    img_path_png = img_dir / f"{base_name}.png"

    if img_path_jpg.exists():
        img_path = img_path_jpg
    elif img_path_png.exists():
        img_path = img_path_png
    else:
        print(f"Error: 找不到影像 {base_name} 的 jpg 或 png 實體檔案。")
        return None, (0, 0)

    # Label 檔案依照 Detectron2 格式，固定為 .png
    lbl_path = Path(root_path) / field_name / 'labels_detectron2' / f"{base_name}.png"

    try:
        # 讀取 RGB 影像
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(img_pil)

        # 讀取 GT 標籤以尋找水花質心
        lbl_np = np.array(Image.open(lbl_path))
        y_indices, x_indices = np.where(lbl_np == 1)

        if len(y_indices) > 0:
            y_center, x_center = int(np.mean(y_indices)), int(np.mean(x_indices))
        else:
            y_center, x_center = img_tensor.shape[1]//2, img_tensor.shape[2]//2 # 若無水花則取中心

        return img_tensor, (x_center, y_center)
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None, (0, 0)

def get_crop_and_bbox(img_tensor, center_coord, crop_size):
    _, H, W = img_tensor.shape
    x_center, y_center = center_coord
    top = max(0, y_center - crop_size // 2)
    left = max(0, x_center - crop_size // 2)
    if top + crop_size > H: top = H - crop_size
    if left + crop_size > W: left = W - crop_size

    patch = TF.crop(img_tensor, top, left, crop_size, crop_size)
    if patch.shape[1] != crop_size or patch.shape[2] != crop_size:
        patch = TF.resize(patch, [crop_size, crop_size])
    return patch, (left, top, left + crop_size, top + crop_size)


# In[10]:


# ================= 3. 主流程：生成視覺儀表板與 JSON =================
# 假設 all_features_norm, labels, kmeans, original_image_paths 仍在記憶體中
cluster_knowledge_draft = []
print(f"--- 開始執行 Visual Context Construction (K={best_k}) ---")

for k in range(best_k):
    print(f"Processing Cluster {k}...")
    indices = np.where(labels == k)[0]
    cluster_vectors = all_features_norm[indices]

    # 計算 Medoid 與 Top-25
    centroid = kmeans.cluster_centers_[k].reshape(1, -1)
    dists_to_centroid = euclidean_distances(cluster_vectors, centroid)
    medoid_local_idx = np.argmin(dists_to_centroid)
    medoid_global_idx = indices[medoid_local_idx]

    dists_to_medoid = euclidean_distances(cluster_vectors, cluster_vectors[medoid_local_idx].reshape(1, -1)).flatten()
    top_m_global_indices = indices[np.argsort(dists_to_medoid)[:25]]

    # 統計資料來源
    source_stats = [FIELD_METADATA_MAPPING.get(original_image_paths[idx]['field'], {}).get('field_id', 'Unknown') for idx in indices]
    source_counts = Counter(source_stats)
    total_count = len(indices)
    dist_str = ", ".join([f"{key}({val/total_count:.1%})" for key, val in source_counts.most_common()])
    dominant_field_id = source_counts.most_common(1)[0][0]
    dominant_meta = next((m for m in FIELD_METADATA_MAPPING.values() if m['field_id'] == dominant_field_id), {})
    avg_score, range_min, range_max = calculate_intensity_stats(source_counts, total_count)

    # 繪製 Medoid Dashboard
    medoid_info = original_image_paths[medoid_global_idx]
    medoid_img_full, medoid_center = read_and_process_image(medoid_info['field'], medoid_info['img_name'])

    if medoid_img_full is not None:
        anchor_patch, bbox = get_crop_and_bbox(medoid_img_full, medoid_center, ANCHOR_SIZE)
        medoid_pil = TF.to_pil_image(medoid_img_full)
        draw = ImageDraw.Draw(medoid_pil)
        draw.rectangle(bbox, outline="red", width=8)
        context_img = TF.to_tensor(medoid_pil.resize((GLOBAL_RESIZE, GLOBAL_RESIZE)))
    else:
        anchor_patch, context_img = torch.zeros(3, ANCHOR_SIZE, ANCHOR_SIZE), torch.zeros(3, GLOBAL_RESIZE, GLOBAL_RESIZE)

    # 繪製 Variance Grid (Top 25)
    variance_patches = []
    for idx in top_m_global_indices:
        var_info = original_image_paths[idx]
        img, center = read_and_process_image(var_info['field'], var_info['img_name'])
        variance_patches.append(get_crop_and_bbox(img, center, PATCH_SIZE)[0] if img is not None else torch.zeros(3, PATCH_SIZE, PATCH_SIZE))

    variance_grid = make_grid(torch.stack(variance_patches), nrow=5, padding=2, normalize=False)
    save_image(variance_grid, os.path.join(OUTPUT_DIR, f"cluster_{k}_grid.png"))
    variance_grid_resized = TF.resize(variance_grid, [GLOBAL_RESIZE, GLOBAL_RESIZE])

    final_dashboard = torch.cat([context_img, TF.resize(anchor_patch, [GLOBAL_RESIZE, GLOBAL_RESIZE]), variance_grid_resized], dim=2)
    dashboard_filename = f"cluster_{k}_dashboard.jpg"
    save_image(final_dashboard, os.path.join(OUTPUT_DIR, dashboard_filename))

    # --- 建構 Knowledge Draft JSON (整合論文最佳化欄位) ---
    cluster_knowledge_draft.append({
        "cluster_id": k,
        "visual_dashboard_path": dashboard_filename,
        "expert_metadata": {
            "dominant_field": dominant_field_id,
            "source_distribution": dist_str,
            "environment_type": dominant_meta.get('environment', 'Unknown'),
            "fish_species": dominant_meta.get('fish_species', 'Unknown'),
            "perspective": dominant_meta.get('perspective', 'Unknown'), # 新增為專家已知資訊
            "splash_intensity": avg_score,
            "intensity_min": range_min,
            "intensity_max": range_max
        },
        "vlm_tasks": {
            # 移除 perspective，保留純視覺推論
            "water_color": "TODO: Select from [dark_blue, greenish, muddy_brown, grey_concrete, black_monochrome]",
            "texture_type": "TODO: Select from [smooth_concentric, chaotic_ripples, foamy_white, glassy, striated_noise]",
            "splash_shape": "TODO: Select from [droplets, columnar, ripple, spray, boiling, chaotic_whitewater, faint_disturbance]",
            "surface_cover": "TODO: Select from [none, white_netting_overlay, floating_algae, foam_scum]",
            "interference": "TODO: Select from [none, paddlewheel_aerator, aerator_bubbles, bird_or_rat, plastic_pipes, green_net_fencing, distant_cages]",
            "lighting": "TODO: Select from [diffuse, high_glare, shadowed, infrared_night_vision, ir_stripes]",
            "container_edge": "TODO: Select from [none, open_water, plastic_edge, concrete_wall, netting]",
            "description": "TODO: Concise visual description"
        }
    })

json_path = os.path.join(OUTPUT_DIR, "cluster_knowledge_draft.json")
with open(json_path, "w", encoding='utf-8') as f:
    json.dump(cluster_knowledge_draft, f, indent=4, ensure_ascii=False)

print(f"\n--- 階段二完成 ---")
print(f"視覺儀表板與已擴充 VLM 屬性之 JSON 成功產出至 {OUTPUT_DIR}")


# In[11]:


import json
import os

# ================= 設定區域 =================
INPUT_DIR = "vlm_knowledge_construction_v3"
INPUT_JSON = os.path.join(INPUT_DIR, "cluster_knowledge_draft.json")

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"錯誤: 找不到 {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding='utf-8') as f:
        data = json.load(f)

    print("========================================================")
    print(f" Gemini 3.0 Advanced 專用 Prompt 生成器 (256D 特徵對齊版)")
    print("========================================================\n")

    for entry in data:
        c_id = entry['cluster_id']
        img_path = entry['visual_dashboard_path']
        meta = entry['expert_metadata']

        avg_intensity = meta.get('splash_intensity', 'Unknown')
        min_int = meta.get('intensity_min', 'Unknown')
        max_int = meta.get('intensity_max', 'Unknown')

        intensity_context = f"Average: {avg_intensity}/10 (Range: {min_int}-{max_int})"

        # [新增] 讀取專家已知的視角資訊
        perspective_context = meta.get('perspective', 'Unknown')

        print(f"### Cluster {c_id} (請上傳圖片: {img_path}) ###")
        print("-" * 20 + " 複製下方文字 " + "-" * 20)

        # 整合了最新論文發現與 256D 物理意義的 Prompt
        prompt = f"""
I am analyzing aquaculture water splash patterns mapped in a 256D Continuous Feature Space. Act as a Computer Vision Expert.

I have provided a "Visual Dashboard" composed of:
1. **LEFT (Context):** A global view (Medoid of the cluster).
2. **CENTER (Anchor):** A close-up prototype of the splash.
3. **RIGHT (Variance):** 25 random sample patches from this specific cluster.

---
[HARD CONSTRAINTS] (Expert Metadata - These are GROUND TRUTHS, do not alter them)
* Environment: {meta.get('environment_type')}
* Source Field: {meta.get('dominant_field')} ({meta.get('source_distribution')})
* Fish Species: {meta.get('fish_species')}
* **Camera Perspective: {perspective_context}** * **Calculated Intensity Profile: {intensity_context}**
  (Note: This intensity is statistically derived from field pixel data. Do NOT guess the score, but describe features matching this intensity.)
---

[DOMAIN KNOWLEDGE RULES]
1. **Imaging Modality:**
   - Check for Infrared (black/white/reddish tint, stripes) vs Visible Light.
2. **Color & Environment:**
   - **Offshore/Sea:** Dark blue, Black.
   - **Onshore Ponds:** Greenish (Algae), Muddy Brown, Grey (Concrete).
3. **Visual Texture Mapping:**
   - **High Intensity (7-10):** Look for "Boiling", "Chaotic Whitewater".
   - **Medium Intensity (4-6):** Look for "Droplets", "Spray", "Ripple".
   - **Low Intensity (1-3):** Look for "Glassy Surface", "Faint Disturbance".
4. **Interference Awareness:**
   - Differentiate between "Paddlewheel Aerator" (large mechanical splashes) and "Aerator Bubbles" (fine underwater bubbles).

---
[TASK: VISUAL TEXTURE ANALYSIS]
Observe the Anchor and Variance images. Generate a structured JSON describing this cluster's pure visual semantics.
**You MUST strictly choose from the provided lists.**

```json
{{
    "water_color": "Select one: [dark_blue, greenish, muddy_brown, grey_concrete, black_monochrome]",
    "texture_type": "Select one: [smooth_concentric, chaotic_ripples, foamy_white, glassy, striated_noise]",
    "splash_shape": "Select one: [droplets, columnar, ripple, spray, boiling, chaotic_whitewater, faint_disturbance]",
    "surface_cover": "Select one: [none, white_netting_overlay, floating_algae, foam_scum]",
    "interference": "Select one: [none, paddlewheel_aerator, aerator_bubbles, bird_or_rat, plastic_pipes, green_net_fencing, distant_cages]",
    "lighting": "Select one: [diffuse, high_glare, shadowed, infrared_night_vision, ir_stripes]",
    "container_edge": "Select one: [none, open_water, plastic_edge, concrete_wall, netting]",
    "description": "A concise sentence describing the visual appearance, explicitly mentioning the water texture, lighting, and splash dynamics."
}}
```"""
        print(prompt.strip())
        print("-" * 55)
        print("\n\n")

if __name__ == "__main__":
    main()


# # Graph RAG

# In[1]:


import json
import os

def generate_final_knowledge_graph_json(input_path, output_path):
    """
    讀取 cluster_knowledge_filled.json (已由 Gemini 填寫完畢的資料)，
    提取 expert_metadata 與 vlm_tasks 的關鍵欄位，
    整合至新的 graph_rag 欄位中，並輸出為 cluster_knowledge_final.json。
    """

    # 1. 檢查檔案是否存在
    if not os.path.exists(input_path):
        print(f"錯誤: 找不到輸入檔案 {input_path}")
        return

    # 2. 讀取 JSON 資料
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"成功讀取 {len(data)} 筆 Cluster 資料，開始處理...")

    # 3. 遍歷並處理每一筆 Cluster
    for cluster in data:
        # 取得來源字典 (若無則給空字典避免報錯)
        expert = cluster.get("expert_metadata", {})
        vlm = cluster.get("vlm_tasks", {})

        # 4. 建構 graph_rag 字典 (扁平化屬性，專供 Neo4j 寫入使用)
        graph_rag_data = {
            # --- 來自 expert_metadata 的數據 (領域專家先驗知識) ---
            "dominant_field": expert.get("dominant_field", "Unknown"),
            "environment_type": expert.get("environment_type", "Unknown"),
            # 魚種改為預設空陣列 []，符合 Neo4j Array 屬性規範
            "fish_species": expert.get("fish_species", []),
            "perspective": expert.get("perspective", "Unknown"),  # [新增] 視角欄位
            "splash_intensity": expert.get("splash_intensity", 0),
            "intensity_min": expert.get("intensity_min", 0),
            "intensity_max": expert.get("intensity_max", 0),
            "source_distribution": expert.get("source_distribution", "Unknown"),

            # --- 來自 vlm_tasks 的數據 (Gemini 事後標註的視覺特徵) ---
            "water_color": vlm.get("water_color", "Unknown"),
            "texture_type": vlm.get("texture_type", "Unknown"),
            "splash_shape": vlm.get("splash_shape", "Unknown"),
            "surface_cover": vlm.get("surface_cover", "none"),
            "interference": vlm.get("interference", "none"),
            "lighting": vlm.get("lighting", "Unknown"),
            "container_edge": vlm.get("container_edge", "none"),
            "description": vlm.get("description", "")
        }

        # 5. 將整理好的 graph_rag 插入 Cluster 物件
        cluster["graph_rag"] = graph_rag_data

    # 6. 寫入新檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"處理完成！新檔案已儲存至: {output_path}")
    print("您現在可以直接使用 'graph_rag' 欄位來生成 Neo4j 節點與屬性。")

# --- 執行設定 ---
# 假設您手動貼完 Gemini 答案的檔案命名為 cluster_knowledge_filled.json
input_json = r'vlm_knowledge_construction_v3\cluster_knowledge_filled.json'  # 若您直接覆蓋原檔，請改成這個名稱
output_json = r'vlm_knowledge_construction_v3\cluster_knowledge_final.json'

if __name__ == "__main__":
    generate_final_knowledge_graph_json(input_json, output_json)


# In[2]:


import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

# ================= 設定區域 =================
ROOT_PATH = r"C:\Users\user\PycharmProjects\organized_ripple_4fold"
FIELDS = ['barrel', 'sea', 'LNG', 'seabass_hmh', 'noon_jsj']

# 讀取 K-Means 分群結果
LABELS_INPUT_PATH = "prompt_256D/kmeans_labels.npy" # 請確認您的 numpy 檔存放路徑
OUTPUT_MAPPING = "vlm_knowledge_construction_v3/pt_to_cluster_mapping.json"

def main():
    print("--- 步驟 1: 讀取 K-means Labels ---")
    if not os.path.exists(LABELS_INPUT_PATH):
        print(f"❌ 錯誤：找不到 {LABELS_INPUT_PATH}")
        return

    labels = np.load(LABELS_INPUT_PATH)
    print(f"✅ 成功讀取 Labels，共 {labels.shape[0]} 筆分群結果")

    print("\n--- 步驟 2: 重新對齊特徵檔與計算強度 (Intensity) ---")
    full_mapping = []
    valid_idx = 0  # 用來與 labels 的 index 對齊

    for field in FIELDS:
        feat_dir = Path(ROOT_PATH) / field / 'features_256'
        lbl_dir = Path(ROOT_PATH) / field / 'labels_detectron2'

        if not feat_dir.exists() or not lbl_dir.exists():
            continue

        pt_files = sorted(list(feat_dir.glob('*.pt')))

        for pt_path in pt_files:
            lbl_path = lbl_dir / f"{pt_path.stem}.png"
            if not lbl_path.exists():
                continue

            # 讀取 Label 來計算水花像素面積 (Intensity)
            try:
                label_img = Image.open(lbl_path)
                label_np = np.array(label_img)
                label_tensor = torch.tensor(label_np).unsqueeze(0).unsqueeze(0).float()

                # Resize 以對齊 200x200 特徵圖空間
                label_resized = F.interpolate(label_tensor, size=(200, 200), mode='nearest').squeeze().long()
                splash_pixel_count = (label_resized == 1).sum().item()

                if splash_pixel_count == 0:
                    continue  # K-means 階段跳過了無水花的影像，此處需保持一致以對齊 labels

                # 建立關聯資訊
                patch_info = {
                    "pt_id": valid_idx,                  # 取代舊的 patch_id
                    "cluster_id": int(labels[valid_idx]),
                    "field": field,
                    "filename": pt_path.name,
                    "pt_absolute_path": str(pt_path.resolve()),  # 供 Stage 2 Dataset 直接讀取
                    "intensity": splash_pixel_count              # 供 Graph RAG 排序挑選
                }
                full_mapping.append(patch_info)
                valid_idx += 1

            except Exception as e:
                print(f"讀取錯誤 {lbl_path}: {e}")

    # 簡單防呆檢查
    if valid_idx != labels.shape[0]:
        print(f"⚠️ 警告：有效特徵筆數 ({valid_idx}) 與 Labels 筆數 ({labels.shape[0]}) 不一致！請檢查資料夾是否有變動。")
        return

    print("\n--- 步驟 3: 儲存 Mapping JSON ---")
    os.makedirs(os.path.dirname(OUTPUT_MAPPING), exist_ok=True)
    with open(OUTPUT_MAPPING, "w", encoding='utf-8') as f:
        json.dump(full_mapping, f, indent=4)

    print(f"✅ 全域索引表已成功建立：{OUTPUT_MAPPING}")
    print(f"總筆數: {len(full_mapping)} 筆 (已從 350萬筆巨量資料成功降維)")
    print("現在您可以正式進入下一步：建立 Neo4j Graph RAG 圖譜。")

if __name__ == "__main__":
    main()


# In[ ]:


import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

# ================= 設定區域 =================
ROOT_PATH = r"C:\Users\user\PycharmProjects\organized_ripple_4fold"
FIELDS = ['barrel', 'sea', 'LNG', 'seabass_hmh', 'noon_jsj']
IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.JPG', '.bmp'] # 支援的原始影像副檔名

# 讀取 K-Means 分群結果
LABELS_INPUT_PATH = "prompt_256D/kmeans_labels.npy"
OUTPUT_MAPPING = "vlm_knowledge_construction_v3/pt_to_cluster_mapping.json"

def main():
    print("--- 步驟 1: 讀取 K-means Labels ---")
    if not os.path.exists(LABELS_INPUT_PATH):
        print(f"❌ 錯誤：找不到 {LABELS_INPUT_PATH}")
        return

    labels = np.load(LABELS_INPUT_PATH)
    print(f"✅ 成功讀取 Labels，共 {labels.shape[0]} 筆分群結果")

    print("\n--- 步驟 2: 重新對齊特徵檔與計算強度 (Intensity) ---")
    full_mapping = []
    valid_idx = 0  # 用來與 labels 的 index 對齊

    for field in FIELDS:
        feat_dir = Path(ROOT_PATH) / field / 'features_256'
        lbl_dir = Path(ROOT_PATH) / field / 'labels_detectron2'
        img_dir = Path(ROOT_PATH) / field / 'images' # 新增：原始影像的目錄路徑

        if not feat_dir.exists() or not lbl_dir.exists() or not img_dir.exists():
            continue

        pt_files = sorted(list(feat_dir.glob('*.pt')))

        for pt_path in pt_files:
            lbl_path = lbl_dir / f"{pt_path.stem}.png"
            if not lbl_path.exists():
                continue

            # 尋找對應的原始影像檔
            image_path = None
            for ext in IMG_EXTENSIONS:
                target_path = img_dir / f"{pt_path.stem}{ext}"
                if target_path.exists():
                    image_path = target_path
                    break

            if image_path is None:
                continue # 若找不到原始影像則跳過此筆資料，確保後續模型讀取不報錯

            # 讀取 Label 來計算水花像素面積 (Intensity)
            try:
                label_img = Image.open(lbl_path)
                label_np = np.array(label_img)
                label_tensor = torch.tensor(label_np).unsqueeze(0).unsqueeze(0).float()

                # Resize 以對齊 200x200 特徵圖空間
                label_resized = F.interpolate(label_tensor, size=(200, 200), mode='nearest').squeeze().long()
                splash_pixel_count = (label_resized == 1).sum().item()

                if splash_pixel_count == 0:
                    continue  # K-means 階段跳過了無水花的影像，此處需保持一致以對齊 labels

                # 建立關聯資訊
                patch_info = {
                    "pt_id": valid_idx,
                    "cluster_id": int(labels[valid_idx]),
                    "field": field,
                    "filename": pt_path.name,
                    "pt_absolute_path": str(pt_path.resolve()),
                    "image_absolute_path": str(image_path.resolve()), # 新增欄位紀錄原始圖片路徑
                    "intensity": splash_pixel_count
                }
                full_mapping.append(patch_info)
                valid_idx += 1

            except Exception as e:
                print(f"讀取錯誤 {lbl_path}: {e}")

    # 簡單防呆檢查
    if valid_idx != labels.shape[0]:
        print(f"⚠️ 警告：有效特徵筆數 ({valid_idx}) 與 Labels 筆數 ({labels.shape[0]}) 不一致！請檢查資料夾是否有變動。")
        return

    print("\n--- 步驟 3: 儲存 Mapping JSON ---")
    os.makedirs(os.path.dirname(OUTPUT_MAPPING), exist_ok=True)
    with open(OUTPUT_MAPPING, "w", encoding='utf-8') as f:
        json.dump(full_mapping, f, indent=4)

    print(f"✅ 全域索引表已成功建立：{OUTPUT_MAPPING}")
    print(f"總筆數: {len(full_mapping)} 筆 (已從 350萬筆巨量資料成功降維)")
    print("現在您可以正式進入下一步：建立 Neo4j Graph RAG 圖譜。")

if __name__ == "__main__":
    main()


# In[3]:


import json
import networkx as nx
import os
import itertools

# ================= 設定區域 =================
INPUT_JSON = "vlm_knowledge_construction_v3/cluster_knowledge_final.json"
GRAPH_OUTPUT_PATH = "vlm_knowledge_construction_v3/knowledge_graph.gml"

# [強度邏輯] 單一魚種強度階級表 (已將多魚種拆分，賦予獨立絕對強度)
SPECIES_RANK = {
    "cobia": 10,                 # 海鱺 (極強)
    "pompano": 8,                # 金鯧 (強)
    "hybrid rock bream": 5,      # 石鯛 (中)
    "seabass": 4,                # 鱸魚 (中弱)
    "black seabream": 3,         # 黑鯛 (弱)
    "seabass fry": 2,            # 鱸魚苗 (極弱)
    "fourfinger threadfin": 1    # 午仔魚 (微弱)
}

def clean_text(text):
    """將底線替換為空白，讓屬性值變成純自然語言，利於 LLM 語義比對"""
    if not isinstance(text, str):
        return str(text)
    return text.replace('_', ' ').strip()

def create_exhaustive_knowledge_graph(json_data):
    G = nx.DiGraph()
    print("正在建構 LLM 友善之知識圖譜 (LLM-Friendly Graph Construction)...")

    existing_species_nodes = set()

    for entry in json_data:
        rag_data = entry.get('graph_rag', {})
        c_id = f"Cluster_{entry['cluster_id']}"

        # ================= 1. Cluster Node (觀測群集) =================
        G.add_node(c_id,
                   type="Cluster",
                   label=f"Cluster {entry['cluster_id']}",
                   description=clean_text(rag_data.get('description', '')),
                   avg_intensity=rag_data.get('splash_intensity', 0),
                   min_intensity=rag_data.get('intensity_min', 0),
                   max_intensity=rag_data.get('intensity_max', 0)
                   )

        # ================= 2. 環境與場域通道 (Metadata) =================
        field_val = clean_text(rag_data.get('dominant_field', 'Unknown'))
        env_val = clean_text(rag_data.get('environment_type', 'Unknown'))
        perspective_val = clean_text(rag_data.get('perspective', 'Unknown'))

        # Field Node
        field_node = f"Field_{field_val.replace(' ', '')}"
        G.add_node(field_node, type="Field", name=field_val)
        G.add_edge(c_id, field_node, relation="ORIGINATES_FROM")

        # Environment Node
        if env_val != "Unknown":
            env_node = f"Env_{env_val.replace(' ', '')}"
            G.add_node(env_node, type="Environment", name=env_val)
            G.add_edge(field_node, env_node, relation="HAS_ENVIRONMENT")

        # Perspective Node
        if perspective_val != "Unknown":
            pers_node = f"Persp_{perspective_val.replace(' ', '')}"
            G.add_node(pers_node, type="Perspective", name=perspective_val)
            G.add_edge(c_id, pers_node, relation="SHOT_FROM")

        # ================= 3. 物種通道 (Fish Species) =================
        # 直接讀取陣列，程式碼大幅簡化
        fish_list = rag_data.get('fish_species', [])
        for fish in fish_list:
            clean_fish_name = clean_text(fish)
            sp_node = f"Species_{clean_fish_name.replace(' ', '')}"
            baseline = SPECIES_RANK.get(clean_fish_name, 0)

            if sp_node not in existing_species_nodes:
                G.add_node(sp_node, type="FishSpecies", name=clean_fish_name, baseline_intensity=baseline)
                existing_species_nodes.add(sp_node)

            # 建立 Cluster -> Species 關聯
            G.add_edge(c_id, sp_node, relation="CONTAINS_SPECIES")

        # ================= 4. Visual Features (視覺特徵) =================
        attributes = {
            'water_color': ('WaterColor', 'HAS_COLOR'),
            'texture_type': ('Texture', 'HAS_TEXTURE'),
            'splash_shape': ('SplashShape', 'HAS_SHAPE'),
            'surface_cover': ('SurfaceCover', 'COVERED_BY'),
            'interference': ('Interference', 'INTERFERED_BY'),
            'lighting': ('Lighting', 'LIT_BY'),
            'container_edge': ('ContainerEdge', 'BOUNDED_BY')
        }

        for json_key, (node_type, rel_name) in attributes.items():
            raw_val = rag_data.get(json_key)
            if raw_val and raw_val != 'none':
                clean_val = clean_text(raw_val)
                feat_node = f"{node_type}_{clean_val.replace(' ', '')}"

                # 節點內部存儲完美的自然語言
                G.add_node(feat_node, type=node_type, name=clean_val)
                G.add_edge(c_id, feat_node, relation=rel_name)

    # ================= 5. 建立物種間強度關係 =================
    print("正在建立魚種強度階級關係 (STRONGER_THAN / WEAKER_THAN)...")
    species_list = list(existing_species_nodes)

    for sp_a, sp_b in itertools.permutations(species_list, 2):
        # 從節點中反查真實的 baseline_intensity
        rank_a = G.nodes[sp_a].get('baseline_intensity', 0)
        rank_b = G.nodes[sp_b].get('baseline_intensity', 0)
        diff = rank_a - rank_b

        if diff > 1:
            G.add_edge(sp_a, sp_b, relation="STRONGER_THAN")
        elif diff < -1:
            G.add_edge(sp_a, sp_b, relation="WEAKER_THAN")
        elif abs(diff) <= 1:
            G.add_edge(sp_a, sp_b, relation="SIMILAR_INTENSITY_TO")

    print(f"圖譜建構完成！節點數: {G.number_of_nodes()}, 邊數: {G.number_of_edges()}")
    return G

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"錯誤: 找不到 {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r", encoding='utf-8') as f:
        data = json.load(f)

    kg = create_exhaustive_knowledge_graph(data)

    os.makedirs(os.path.dirname(GRAPH_OUTPUT_PATH), exist_ok=True)
    nx.write_gml(kg, GRAPH_OUTPUT_PATH)
    print(f"GML 檔案已儲存至: {GRAPH_OUTPUT_PATH}")

if __name__ == "__main__":
    main()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script G.ipynb')

