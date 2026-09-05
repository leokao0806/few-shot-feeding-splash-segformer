import itertools
import numpy as np
import os
import torch
from glob import glob
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from typing import List, Tuple, Union
from pathlib import Path
import random
import torchvision.transforms.functional as TF


# ==========================================
# 1. 原始 RippleDataset (保留，不更動)
# ==========================================
class RippleDataset(Dataset):
    CLASSES = ['unlabel', 'ripple']
    PALETTE = torch.tensor([
        [120, 120, 120], [41, 255, 0]
    ])

    def __init__(self, root: str, field: List[str], split: str = 'train', transform=None, k=0, sample_num=0):
        super().__init__()
        assert split in ['train', 'val']
        self.split = 'training' if split == 'train' else 'validation'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.root = root
        self.field = field
        self.k = k
        self.sample_num = sample_num

        self.files, self.field_ids = self.get_ripple_data_paths(imgs_only=False, k=k)

    def get_ripple_data_paths(self, imgs_only=False, k=0) -> Union[List[str], Tuple[List[str], List[int]]]:
        fields_num = len(self.field)
        field_img_names = [None] * fields_num
        for i, field in enumerate(self.field):
            for j in range(1, 5):
                if j == k + 1:
                    continue
                field_path = os.path.join(self.root, field, f'fold_{j}')
                each_fold = glob(os.path.join(field_path, 'images', '*.png')) + glob(
                    os.path.join(field_path, 'images', '*.jpg'))
                field_img_names[i] = each_fold if field_img_names[i] is None else field_img_names[i] + each_fold
            assert len(field_img_names[i]) > 0, f'No images found in {field_path}'

        min_len = min([len(field_img) for field_img in field_img_names])
        validation_split = 0.1
        indices = np.arange(min_len)
        np.random.shuffle(indices)
        split = int(np.floor(validation_split * min_len))
        train_indices, val_indices = indices[split:], indices[:split]

        for i in range(fields_num):
            if self.split == 'training':
                current_train_indices = train_indices
                if self.sample_num > 0:
                    current_train_indices = current_train_indices[:self.sample_num]
                field_img_names[i] = [field_img_names[i][j] for j in current_train_indices]
            elif self.split == 'validation':
                field_img_names[i] = [field_img_names[i][j] for j in val_indices]

        files_len = len(field_img_names[0])
        imgs = list(itertools.chain.from_iterable(zip(*[field_img_names[i][:files_len] for i in range(fields_num)])))
        imgs_field_id = list(itertools.chain.from_iterable(zip(*[np.full(files_len, i) for i in range(fields_num)])))

        if imgs_only:
            return imgs
        return imgs, imgs_field_id

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, int]:
        image_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'labels_detectron2')
        field_id = self.field_ids[index]

        img = io.read_image(image_path, io.ImageReadMode.RGB)
        label = io.read_image(lbl_path, io.ImageReadMode.GRAY)

        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label.squeeze().long(), field_id


# ==========================================
# 2. 輔助函式
# ==========================================
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.png', '.bmp']


def find_source_image(label_path):
    label_dir = os.path.dirname(label_path)
    filename = os.path.basename(label_path)
    name_no_ext = os.path.splitext(filename)[0]
    clean_name = name_no_ext[:-5] if name_no_ext.endswith('_mask') else name_no_ext

    search_dirs = [label_dir, os.path.join(os.path.dirname(label_dir), 'image')]
    for directory in search_dirs:
        if not os.path.exists(directory): continue
        for ext in IMG_EXTENSIONS:
            target_path = os.path.join(directory, clean_name + ext)
            if os.path.exists(target_path):
                return target_path
    return None


# ==========================================
# 3. 升級版 RippleFeatureDataset (Stage 2 核心)
# ==========================================
class RippleFeatureDataset(Dataset):
    """
    支援讀取外部排除清單 (txt) 的進階 Dataset。
    確保測試集 (Pure Test) 與訓練/驗證集 (Few-shot Pool) 完全互斥。
    """

    def __init__(self, root: str, field: list = None, split: str = 'train', sample_num=20,
                 val_size=5, log_dir: str = None, unseen_map: dict = None,
                 seed=3402, is_pure_test=False, list_dir: str = None,
                 test_limit: int = None, graph_rag_image_list: list = None):
        super().__init__()
        self.root = root
        self.split = split
        self.ignore_label = 255
        self.n_classes = 2
        self.files = []
        self.is_unseen_domain = False
        self.sample_num = sample_num  # 這是總標注數 K
        self.val_size = val_size
        self.unseen_map = unseen_map or {}
        self.is_pure_test = is_pure_test
        self.list_dir = list_dir  # 記錄排除清單的路徑
        self.test_limit = test_limit  # 儲存限制數量

        # 初始化隨機數產生器 (僅用於無清單時的打亂)
        rng = random.Random(seed)

        if field is None:
            raise ValueError("❌ 錯誤: 必須指定 field 參數。")

        # ============================================================
        # 模式 C: 未見場域 (F~J) - 處理原圖
        # ============================================================
        if field[0] in self.unseen_map:
            self.is_unseen_domain = True
            domain_key = field[0]
            label_dir = Path(self.unseen_map[domain_key])
            all_labels = sorted(list(label_dir.glob('*.png')))

            # 建立完整的 (Image, Label) 配對池
            full_paired_pool = []
            for lbl_path in all_labels:
                img_path = find_source_image(str(lbl_path))
                if img_path:
                    full_paired_pool.append((img_path, str(lbl_path)))

            # --- 核心邏輯：處理測試集的排除 ---
            if self.list_dir:
                if self.is_pure_test:
                    # 1. 讀取該場域已記錄的排除檔名 (從 train 和 val 清單)
                    exclude_names = self._load_exclude_names(domain_key)
                    # 2. 剔除在清單內的檔案
                    self.files = [p for p in full_paired_pool if os.path.basename(p[0]) not in exclude_names]
                    # 實施測試數量限制
                    if self.test_limit is not None:
                        self.files = full_paired_pool[:self.test_limit]

                    print(
                        f">>> [Standardized Test] {domain_key}: 讀取清單排除 {len(exclude_names)} 張，剩餘 {len(self.files)} 張作為測試集。")
                else:
                    self.files = self._load_files_from_split_list(domain_key, self.split, full_paired_pool)
                    # --- 增加明確的讀取數量輸出 ---
                    split_label = "Training" if self.split == 'train' else "Validation"
                    print(f">>> [Fixed {split_label}] Domain {domain_key}: 從清單載入 {len(self.files)} 張樣本。")
            else:
                # --- 模式：建立新實驗 (隨機打亂並劃分) ---
                rng.shuffle(full_paired_pool)
                few_shot_pool = full_paired_pool[:self.sample_num]
                test_pool = full_paired_pool[self.sample_num:]

                if self.is_pure_test:
                    self.files = test_pool
                elif self.split == 'val':
                    self.files = few_shot_pool[:self.val_size]
                else:
                    self.files = few_shot_pool[self.val_size:]

                # 如果有設定 log_dir，則儲存目前的清單 (以便之後跑測試時讀取)
                if log_dir:
                    self._save_sample_list(log_dir, domain_key)

                print(f">>> [Dataset C] {domain_key} ({self.split}): K={self.sample_num}, 分配 {len(self.files)} 張。")

        # ============================================================
        # 模式 A: 已知場域 (A~E)
        # ============================================================
        else:
            if not graph_rag_image_list:
                raise ValueError(
                    f"❌ 錯誤: {field} 屬於已知場域，必須透過 graph_rag_image_list 傳入 Graph RAG 檢索結果！")

            # 1. 建立完整的 Graph RAG 檢索資料池
            rag_pool = []
            for img_path_str in graph_rag_image_list:
                img_p = Path(img_path_str)
                # 推導對應的 Label 路徑 (將 images 資料夾替換為 labels_detectron2，副檔名強制為 .png)
                lbl_path = img_p.parents[1] / 'labels_detectron2' / f"{img_p.stem}.png"

                if lbl_path.exists():
                    rag_pool.append((str(img_p), str(lbl_path)))
                else:
                    print(f"⚠️ 警告: 找不到對應的標註檔 {lbl_path}")

            # 2. 進行打亂與資料集切分 (以確保 Q prime 微調的隨機性與評估客觀性)
            # 使用已初始化的 rng (random.Random(seed)) 確保每次執行的打亂順序一致
            rng.shuffle(rag_pool)

            # 3. 根據 split 狀態分配資料
            # 假設 sample_num 為 20，val_size 為 5，則 train 為前 15 張，val 為後 5 張
            train_size = self.sample_num - self.val_size

            if self.split == 'train':
                self.files = rag_pool[:train_size]
            elif self.split == 'val':
                self.files = rag_pool[train_size:self.sample_num]
            else:
                self.files = rag_pool  # 容錯機制，保留全拿的可能性

            print(
                f">>> [Dataset A (Graph RAG)] {self.split.upper()} 模式: 成功載入 {len(self.files)} 張 Prompt 影像，準備提供 Q prime 特徵拼裝。")

    def _load_files_from_split_list(self, domain_key, split, full_paired_pool):
        """
        新增：根據 split (train/val) 讀取對應清單，並從 pool 中提取完整路徑。
        """
        target_names = set()
        list_file = Path(self.list_dir) / f"{domain_key}_mask_hard_samples_val_list.txt"
        #list_file = Path(self.list_dir) / f"{domain_key}_seed_K20_val_list.txt"

        if list_file.exists():
            with open(list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if "IMG: " in line:
                        name = line.split('|')[0].replace("IMG: ", "").strip()
                        target_names.add(name)
        else:
            print(f"⚠️ 錯誤: 找不到清單 {list_file}")

        # 比對 pool，回傳完整路徑的 tuple list
        return [p for p in full_paired_pool if os.path.basename(p[0]) in target_names]

    # def _load_exclude_names(self, domain_key):
    #     """
    #     從 list_dir 中讀取對應場域的 train 與 val 清單，提取所有檔名。
    #     """
    #     exclude_set = set()
    #     # 同時讀取 train 和 val 清單，因為這兩者構成了完整的 K=20 池子
    #     for split_type in ['train', 'val']:
    #         list_file = Path(self.list_dir) / f"{domain_key}_seed_K{self.sample_num}_{split_type}_list.txt"
    #         if list_file.exists():
    #             with open(list_file, 'r', encoding='utf-8') as f:
    #                 for line in f:
    #                     if "IMG: " in line:
    #                         # 解析格式 "IMG: sample_01.jpg | PATH: ..."
    #                         name = line.split('|')[0].replace("IMG: ", "").strip()
    #                         exclude_set.add(name)
    #         else:
    #             print(f"⚠️ 警告: 找不到清單檔案 {list_file.name}，請檢查路徑。")
    #     return exclude_set
    def _load_exclude_names(self, domain_key):
        """
        從 list_dir 中讀取對應場域的 val 清單，提取所有檔名。
        目前僅排除 MIL 使用的 5 張 val 提示，將未被使用的 15 張 train 放回測試集。
        """
        exclude_set = set()

        # 直接指定讀取 val 清單即可
        list_file = Path(self.list_dir) / f"{domain_key}_mask_hard_samples_val_list.txt"
        #list_file = Path(self.list_dir) / f"{domain_key}_seed_K20_val_list.txt"

        if list_file.exists():
            with open(list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if "IMG: " in line:
                        # 解析格式 "IMG: sample_01.jpg | PATH: ..."
                        name = line.split('|')[0].replace("IMG: ", "").strip()
                        exclude_set.add(name)
        else:
            print(f"⚠️ 警告: 找不到清單檔案 {list_file.name}，請檢查路徑。")

        return exclude_set

    def _save_sample_list(self, log_dir, field_name):
        save_path = Path(log_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        list_file = save_path / f"{field_name}_seed_K{self.sample_num}_{self.split}_list.txt"

        with open(list_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment: {field_name} | K_Total: {self.sample_num} | Split: {self.split}\n")
            f.write("-" * 60 + "\n")
            for img_path, lbl_path in self.files:
                f.write(f"IMG: {os.path.basename(img_path)} | PATH: {img_path}\n")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_path, lbl_path = self.files[index]
        lbl = io.read_image(lbl_path, io.ImageReadMode.GRAY)
        lbl = TF.resize(lbl, [800, 800], interpolation=TF.InterpolationMode.NEAREST).squeeze()
        # 統一處理 Image (不論新舊場域，現在都是讀取原始影像)
        img = io.read_image(data_path, io.ImageReadMode.RGB)
        img = TF.resize(img, [800, 800], interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.convert_image_dtype(img, torch.float)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img, lbl.long()