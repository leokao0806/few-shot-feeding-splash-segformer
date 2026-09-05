import torch
import os
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader

from util.analysis_algs import MultiInstanceLearning, solve_optimal_alpha, MultiInstanceLearning_calculate_w
from util.features_core import (
    get_roi_instances,
    find_roi_by_bbox,
    extract_point_prompt_instances,
    extract_mask_positive_instances,
)
from dataset import RippleFeatureDataset


class MilPipeline:
    def __init__(self, model, device, dataset_cfg=None, totalsubclass=64, weight_dir='Subclass/cache'):
        self.model = model
        self.device = device
        self.dataset_cfg = dataset_cfg
        self.totalsubclass = totalsubclass
        self.weight_dir = Path(weight_dir)
        self.weight_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_weight = None

    def offline_compute_and_save_weights(
        self,
        domain_key,
        method='box',
        need_negative=True,
        target_n_num=None,
        list_dir=None,
        support_shots=None,   # ← 新增：實際 support 張數，None 時退回舊行為
    ):
        if self.dataset_cfg is None:
            raise ValueError("提取特徵與計算權重需要提供 dataset_cfg！")

        print(f">>> [Mode: {method.upper()}] 正在從場域 {domain_key} 提取提示並計算權重...")
        self.model.eval()

        current_list_dir = list_dir if list_dir is not None else self.dataset_cfg.get('LIST_DIR', 'Subclass/exclude')

        # 修正：support_shots 未指定時才退回 sample_num=20 的舊行為
        effective_sample_num = support_shots if support_shots is not None else 20

        dataset = RippleFeatureDataset(
            root=self.dataset_cfg['ROOT'],
            field=[domain_key],
            split='val',
            sample_num=effective_sample_num,
            unseen_map=self.dataset_cfg.get('UNSEEN_DOMAINS', {}),
            list_dir=current_list_dir,
            is_pure_test=False,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        print(f"\n🔍 [MIL 提示診斷] 場域 {domain_key} 實際使用的提示圖片 (共 {len(dataset)} 張):")
        for img_path, _ in dataset.files:
            print(f"   👉 {os.path.basename(img_path)}")
        print("-" * 50 + "\n")

        if len(dataset) == 0:
            raise ValueError(
                f"❌ 場域 {domain_key} 的 MIL dataset 是空的！\n"
                f"   請確認 list_dir={current_list_dir} 底下有對應的清單檔，"
                f"且清單內的影像存在於 unseen_map 路徑中。"
            )

        p_instances_list = []
        n_instances_list = []

        with torch.no_grad():
            for img, lbl in loader:
                img, lbl = img.to(self.device), lbl.to(self.device)
                _, subclass_map = self.model(img)

                p_inst, n_inst = None, None
                if method == 'box':
                    mask_200 = F.interpolate(
                        lbl.unsqueeze(0).float(), size=(200, 200), mode='nearest'
                    ).squeeze().cpu().numpy()
                    _, roi_box, _ = find_roi_by_bbox(mask_200, margin_px=20)
                    p_inst, n_inst = get_roi_instances(subclass_map, roi_box, need_negative=need_negative)

                elif method == 'point':
                    p_inst, n_inst = extract_point_prompt_instances(
                        subclass_map, lbl, kernel_size=7, need_negative=need_negative
                    )

                elif method == 'mask':
                    p_inst, n_inst = extract_mask_positive_instances(
                        subclass_map, lbl, need_negative=need_negative, target_n_num=target_n_num
                    )

                if p_inst is not None and len(p_inst) > 0:
                    p_instances_list.append(p_inst)
                if n_inst is not None and len(n_inst) > 0:
                    n_instances_list.append(n_inst)

        if not p_instances_list:
            raise ValueError(f"場域 {domain_key} 使用 {method} 模式未提取到任何正樣本！")

        all_p = torch.cat(p_instances_list, dim=0).cpu()
        all_n = torch.cat(n_instances_list, dim=0).cpu() if n_instances_list else None

        # 從模型中提取 Vb（領域知識庫）
        Vb = None
        for name, param in self.model.named_parameters():
            if 'Vb' in name:
                Vb = param.detach().cpu().numpy()
                break

        if Vb is None:
            raise ValueError("無法在模型參數中找到 Vb！請確認模型架構。")

        optimized_weight = MultiInstanceLearning_calculate_w(all_p, all_n, Vb)

        # alpha路徑(linear programming) 把上面註解 下面兩行取消註解
        # dealt = MultiInstanceLearning(all_p, all_n)
        # optimized_weight = solve_optimal_alpha(dealt, Vb)

        prefix = "mil_weight" if need_negative else "mil_weight_no_n"
        out_file = self.weight_dir / f'{prefix}_{method}_{domain_key}.pt'
        torch.save(optimized_weight, out_file)
        print(f"✅ [{method}] 權重存檔: {out_file} | Shape: {optimized_weight.shape}")

        return optimized_weight

    def load_weights(self, domain_key, method='box', need_negative=True):
        prefix = "mil_weight" if need_negative else "mil_weight_no_n"
        weight_path = self.weight_dir / f'{prefix}_{method}_{domain_key}.pt'

        if weight_path.exists():
            print(f"✅ 載入已固定的少樣本提示權重 ({method}): {weight_path}")
            self.fixed_weight = torch.load(weight_path, map_location=self.device)
            return self.fixed_weight
        else:
            raise FileNotFoundError(f"[!] 未發現權重檔: {weight_path}")

    def get_inference_weight(self):
        if self.fixed_weight is None:
            raise RuntimeError("權重尚未載入！請先呼叫 load_weights。")
        return self.fixed_weight