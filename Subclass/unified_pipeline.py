import os
import sys
from pathlib import Path

# 環境與路徑配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import yaml
import torch
import numpy as np
import copy
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt

# 核心模型與資料集
from semseg.models.subclass_segformer_unified import SubclassSegFormer_Unified
from dataset import RippleFeatureDataset
from semseg.utils.utils import fix_seeds

# 導入自定義 Pipeline 模組
from prompt.core_pipeline import AquacultureGraphRAG
from mil_pipeline import MilPipeline
from qprime_pipeline import QprimePipeline


class UnifiedInferencePipeline:
    def __init__(self, cfg, device, mil_method='box'):
        print("✅ 開始初始化 Pipeline...", flush=True)
        self.cfg = cfg
        self.device = device
        self.model_cfg = cfg['MODEL']
        self.dataset_cfg = cfg['DATASET']
        self.prompt_cfg = cfg.get('PROMPT_LEARNING', {})
        self.mil_method = mil_method

        # 1. 初始化模型
        print("✅ 正在載入模型結構與權重...", flush=True)
        self.model = self._load_model()

        # 2. 初始化 Graph RAG
        self.mapping_path = r"C:\Users\user\PycharmProjects\subclass_segformer\Subclass\prompt\vlm_knowledge_construction_v3\pt_to_cluster_mapping.json"
        self.knowledge_path = r"C:\Users\user\PycharmProjects\subclass_segformer\Subclass\prompt\vlm_knowledge_construction_v3\cluster_knowledge_final.json"
        self.rag_engine = None

        # 3. 初始化 MIL 專家 (傳入 dataset_cfg 支援離線特徵提取)
        print("✅ 正在初始化 MIL Engine...", flush=True)
        self.mil_engine = MilPipeline(
            model=self.model,
            device=self.device,
            dataset_cfg=self.dataset_cfg,
            weight_dir='Subclass/cache'
        )

        # 4. 初始化 Q-prime 專家
        print("✅ 正在初始化 Q-prime Pipeline...", flush=True)
        self.q_tuner = QprimePipeline(self.cfg, self.model, self.device)

        # 5. 備份 Stage 1 原始權重 (用於 Q-prime 微調前的重置)
        self.base_q_prime = self.model.decode_head.subclass_block.q_prime.detach().clone()
        self.base_gamma = self.model.decode_head.subclass_block.gamma.detach().clone()
        # 6. 新增：Q-prime 記憶體快取 (避免 Phase 2 重複訓練)
        self.tuned_q_prime = None
        # ⬇️ 新增：快取微調後的 gamma
        self.tuned_gamma = None
        self.cached_llm_intent = {}

        print("[Debug] 初始化全部完成。", flush=True)

    def _load_model(self):
        model = SubclassSegFormer_Unified(
            backbone=self.model_cfg['BACKBONE'],
            num_classes=2,
            num_prompts=self.prompt_cfg.get('NUM_PROMPTS', 5)
        )
        pretrained_path = self.model_cfg.get('PRETRAINED', None)
        if pretrained_path and Path(pretrained_path).exists():
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint,
                                                                                dict) else checkpoint.state_dict()
            # ⬇️ 新增：過濾掉維度不匹配的權重，避免 load_state_dict 崩潰
            q_prime_key = 'decode_head.subclass_block.q_prime'
            if q_prime_key in state_dict:
                ckpt_shape = state_dict[q_prime_key].shape
                model_shape = model.decode_head.subclass_block.q_prime.shape
                if ckpt_shape != model_shape:
                    print(
                        f"⚠️ [權重過濾] 發現 q_prime 維度不匹配 (ckpt: {ckpt_shape} -> model: {model_shape})。已丟棄舊權重，使用隨機初始化！")
                    del state_dict[q_prime_key]
            # ⬆️ 新增結束
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ 模型權重載入完成: {pretrained_path}")
        return model.to(self.device)

    def calculate_iou(self, pred_logits, gt_tensor):
        pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0)
        intersect = torch.logical_and(pred_mask == 1, gt_tensor == 1).sum().item()
        union = torch.logical_or(pred_mask == 1, gt_tensor == 1).sum().item()
        return intersect / (union + 1e-6)

    def calculate_precision(self, pred_logits, gt_tensor):
        """新增：計算 Precision (TP / (TP + FP))"""
        pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0)
        tp = torch.logical_and(pred_mask == 1, gt_tensor == 1).sum().item()
        fp = torch.logical_and(pred_mask == 1, gt_tensor == 0).sum().item()
        return tp / (tp + fp + 1e-6)

    def _save_mask_overlay(self, img_tensor, gt_tensor, pred_tensor, save_path):
        """
        將原圖與 Ground Truth (左)、原圖與 Prediction (右) 並排對比儲存。
        遮罩統一使用透明紅色。
        """
        # 影像反正規化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
        img = img_tensor * std + mean
        img = img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

        gt = gt_tensor.cpu().numpy()
        pred = pred_tensor.cpu().numpy()

        h, w = img.shape[:2]

        # 建立畫布：1 橫列，2 直行
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # ==================== 左圖：Ground Truth ====================
        axes[0].imshow(img)
        gt_rgba = np.zeros((h, w, 4))
        gt_rgba[gt == 1] = [1, 0, 0, 0.5]  # [R, G, B, Alpha]，統一用透明紅色
        axes[0].imshow(gt_rgba)
        axes[0].set_title("Ground Truth", fontsize=16)
        axes[0].axis('off')

        # ==================== 右圖：Prediction ====================
        axes[1].imshow(img)
        pred_rgba = np.zeros((h, w, 4))
        pred_rgba[pred == 1] = [1, 0, 0, 0.5]  # [R, G, B, Alpha]，統一用透明紅色
        axes[1].imshow(pred_rgba)
        axes[1].set_title("Prediction", fontsize=16)
        axes[1].axis('off')

        # 調整排版並存檔
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    def _extract_and_save_hard_samples(self, results_info, field, k=5):
        """
        提取最低分的 k 張樣本，依照讀取格式輸出至 txt，並將這 k 張的 Mask 儲存為疊圖。
        """
        # 依據 IoU 遞增排序，抓取最低分
        sorted_info = sorted(results_info, key=lambda x: x['iou'])
        hardest_samples = sorted_info[:k]

        save_dir = Path("Subclass/debug_logs")
        save_dir.mkdir(parents=True, exist_ok=True)

        # txt 命名也加上 mil_method 避免被覆蓋
        txt_path = save_dir / f"{field}_{self.mil_method}_hard_samples_val_list.txt"

        # 建立 Mask 專屬資料夾 (加上 mil_method 區別)
        mask_save_dir = save_dir / f"masks_{field}_{self.mil_method}"
        mask_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"🔍 診斷模式：最低分 {k} 張 Mask 將儲存於 {mask_save_dir}")

        with open(txt_path, 'w', encoding='utf-8') as f:
            for item in hardest_samples:
                img_path = item['path']
                basename = os.path.basename(img_path)
                iou = item['iou']

                # 寫入 txt
                f.write(f"IMG: {basename} | PATH: {img_path}\n")

                # 輸出疊圖 (僅針對這 5 張)
                mask_filename = mask_save_dir / f"iou_{iou:.4f}_{basename}"
                self._save_mask_overlay(
                    item['img_tensor'],
                    item['gt_tensor'],
                    item['pred_tensor'],
                    mask_filename
                )

        print(f"📁 困難樣本清單與疊圖已儲存完畢。")

    def _prepare_rag_tuning_data(self, field, prompt_images):
        """
        專門為 Q-prime Tuning 準備資料：使用 RAG 抓回來的影像。
        依據實際檢索回傳的總量動態切分訓練與驗證集 (例如保留 25% 作為驗證)。
        """
        total_samples = len(prompt_images)
        dynamic_val_size = max(1, int(total_samples * 0.25))  # 動態計算驗證集大小

        train_set = RippleFeatureDataset(
            root=self.dataset_cfg['ROOT'],
            field=[field],
            split='train',
            sample_num=total_samples,
            val_size=dynamic_val_size,
            graph_rag_image_list=prompt_images
        )

        val_set = RippleFeatureDataset(
            root=self.dataset_cfg['ROOT'],
            field=[field],
            split='val',
            sample_num=total_samples,
            val_size=dynamic_val_size,
            graph_rag_image_list=prompt_images
        )

        train_loader = DataLoader(train_set, batch_size=self.prompt_cfg.get('BATCH_SIZE', 5), shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        return train_loader, val_loader

    def _prepare_adaptation_data(self, field):
        """
        專供推論階段使用，僅回傳 Unseen Domain (F~J) 的 Test Loader。
        嚴格斷開與 Tuning 階段的任何關聯。
        """
        k_shot = self.dataset_cfg.get('SAMPLE_PER_FIELD', 20)
        seed = 3402
        list_dir_cfg = self.dataset_cfg.get('LIST_DIR', 'Subclass/exclude')

        test_set = RippleFeatureDataset(
            root=self.dataset_cfg['ROOT'],
            field=[field],
            is_pure_test=True,
            sample_num=k_shot,
            seed=seed,
            list_dir=list_dir_cfg,
            unseen_map=self.dataset_cfg.get('UNSEEN_DOMAINS')
        )

        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        return test_loader

    def run_baseline_inference(self, test_loader):
        self.model.eval()
        ious = []
        precs = []
        with torch.no_grad():
            for img, lbl in tqdm(test_loader, desc="[Baseline] Inference"):
                img, lbl = img.to(self.device), lbl.to(self.device)
                output, _ = self.model(img, q_prime=False)
                ious.append(self.calculate_iou(output, lbl.squeeze(0)))
                precs.append(self.calculate_precision(output, lbl.squeeze(0)))
        # 回傳形狀 (N, 2)，Column 0 是 IoU，Column 1 是 Precision
        return np.stack((ious, precs), axis=-1)

    def run_mil_inference(self, test_loader, field='', diagnostic_mode=False):
        self.model.eval()
        ious = []
        precs = []
        results_info = []

        fixed_weight = self.mil_engine.get_inference_weight().to(self.device)

        for idx, (img, lbl) in enumerate(tqdm(test_loader, desc="[MIL] Inference")):
            img, lbl = img.to(self.device), lbl.to(self.device)
            with torch.no_grad():
                output, _ = self.model(img, subclass_weight=fixed_weight)

                if output.shape[2:] != lbl.shape[1:]:
                    output = F.interpolate(
                        output,
                        size=lbl.shape[1:],
                        mode='bilinear',
                        align_corners=False
                    )

                iou = self.calculate_iou(output, lbl.squeeze(0))
                prec = self.calculate_precision(output, lbl.squeeze(0))
                ious.append(iou)
                precs.append(prec)

                if diagnostic_mode:
                    data_path, _ = test_loader.dataset.files[idx]
                    pred_mask = torch.argmax(output, dim=1).squeeze(0)

                    results_info.append({
                        'path': data_path,
                        'iou': iou,
                        'img_tensor': img.squeeze(0).cpu(),
                        'gt_tensor': lbl.squeeze(0).cpu(),
                        'pred_tensor': pred_mask.cpu()
                    })

        if diagnostic_mode:
            self._extract_and_save_hard_samples(results_info, field=field, k=5)

        return np.stack((ious, precs), axis=-1)

    def run_qprime_inference(self, field, text_prompt, skip_tuning=False, total_n=20):
        if skip_tuning and self.tuned_q_prime is not None:
            print("⚡ [Phase 2 加速] 偵測到已在 Phase 1 完成 Q-prime 微調，直接沿用最佳權重！")
            self.model.decode_head.subclass_block.q_prime.data = self.tuned_q_prime.to(self.device)
            if self.tuned_gamma is not None:
                self.model.decode_head.subclass_block.gamma.data = self.tuned_gamma.to(self.device)
            llm_intent = self.cached_llm_intent
            prompt_images = ["Cached"]
        else:
            if self.rag_engine is None:
                print("✅ 正在初始化 Graph RAG 引擎與圖譜連線...", flush=True)
                self.rag_engine = AquacultureGraphRAG(
                    pt_mapping_path=self.mapping_path,
                    knowledge_json_path=self.knowledge_path,
                    neo4j_pw="66386638"
                )

            print(f"🔍 [Step 1] Graph RAG 檢索中: {text_prompt}")
            rag_results = self.rag_engine.query(text_prompt, total_n=total_n)
            prompt_images = rag_results["image_list"]
            llm_intent = rag_results["intent"]

            if not prompt_images:
                print("❌ RAG 未命中，跳過微調。")
                return None, llm_intent

            train_loader, val_loader = self._prepare_rag_tuning_data(field, prompt_images)
            self.model.decode_head.subclass_block.q_prime.data = self.base_q_prime.data.to(self.device)
            self.model.decode_head.subclass_block.gamma.data = self.base_gamma.data.to(self.device)
            self.q_tuner.fit(train_loader, val_loader, field)

        test_loader = self._prepare_adaptation_data(field)

        self.model.eval()
        ious = []
        precs = []
        with torch.no_grad():
            for img, lbl in tqdm(test_loader, desc=f"[Q-prime] Inference {field}"):
                img, lbl = img.to(self.device), lbl.to(self.device)
                output, _ = self.model(img, q_prime=True)

                if output.shape[2:] != lbl.shape[1:]:
                    output = F.interpolate(output, size=lbl.shape[1:], mode='bilinear', align_corners=False)

                ious.append(self.calculate_iou(output, lbl.squeeze(0)))
                precs.append(self.calculate_precision(output, lbl.squeeze(0)))

        return np.stack((ious, precs), axis=-1), llm_intent

    def execute(self, method, field, file_name='', text_prompt='', skip_tuning=False, need_negative=True, enable_diagnostics=False):
        print(f"\n🚀 啟動實驗 | 方法: {method} | 場域: {field}")
        fix_seeds(3402)

        results = None
        current_method = method.lower()

        if current_method == 'qprime':
            if not text_prompt:
                raise ValueError(f"❌ 錯誤：執行 qprime 方法必須提供對應的 text_prompt。")
            results, llm_intent = self.run_qprime_inference(field, text_prompt, skip_tuning=skip_tuning)

            if results is None or len(results) == 0:
                print(f"⚠️ [Fallback] Graph RAG 檢索未命中，系統退避至 Baseline 模式。")
                current_method = 'baseline'
                file_name = f"{field}_fallback"

        if current_method in ['baseline', 'mil']:
            test_loader = self._prepare_adaptation_data(field)

            if current_method == 'baseline':
                print("🔍 執行 Baseline 推論...")
                results = self.run_baseline_inference(test_loader)

            elif current_method == 'mil':
                try:
                    self.mil_engine.load_weights(domain_key=field, method=self.mil_method, need_negative=need_negative)
                except FileNotFoundError:
                    print(f"⚠️ 尚未建立場域 {field} 的 MIL 權重，啟動離線計算流程...")
                    self.mil_engine.offline_compute_and_save_weights(domain_key=field, method=self.mil_method,
                                                                     need_negative=need_negative)
                    self.mil_engine.load_weights(domain_key=field, method=self.mil_method, need_negative=need_negative)

                print("🔍 執行 MIL 推論...")
                results = self.run_mil_inference(test_loader, field=field, diagnostic_mode=enable_diagnostics)

        if results is not None and len(results) > 0:
            save_dir = Path("Subclass/unified_npy")
            save_dir.mkdir(exist_ok=True)

            method_suffix = f"{current_method}_{self.mil_method}" if current_method == 'mil' else current_method
            # 檔案命名可改為 metrics_ 開頭表示裡面有多個指標
            base_filename = f"metrics_unified{file_name}_{method_suffix}_domain_{field}"

            npy_path = save_dir / f"{base_filename}.npy"
            np.save(npy_path, results)

            # 從 2D 陣列中提取平均值
            mean_iou = float(np.mean(results[:, 0]))
            mean_prec = float(np.mean(results[:, 1]))

            print(f"✅ 實驗完成。平均 IoU: {mean_iou:.4f} | 平均 Precision: {mean_prec:.4f}")
            print(f"💾 NPY 存檔於: {npy_path} (陣列形狀: {results.shape}，[IoU, Precision])")

            if method.lower() == 'qprime':
                log_data = {
                    "field": field,
                    "method": current_method,
                    "mean_iou": mean_iou,
                    "mean_precision": mean_prec,
                    "timestamp": datetime.now().isoformat(),
                    "text_prompt": text_prompt,
                    "llm_intent": llm_intent
                }
                json_path = save_dir / f"{base_filename}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=4)
                print(f"💾 JSON 日誌存檔於: {json_path}")
        else:
            print("❌ 實驗未能產生有效結果。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ripple_prompt.yaml')
    parser.add_argument('--file_name', type=str, default='')
    parser.add_argument('--field', type=str, required=True)
    parser.add_argument('--method', type=str, default='mil', choices=['baseline', 'mil', 'qprime'])
    parser.add_argument('--mil_method', type=str, default='bag', choices=['box', 'point', 'mask'],
                        help="指定提示策略：box (框提示), point (點提示), mask (完美正樣本)")
    parser.add_argument('--text_prompt', type=str, default='')
    parser.add_argument('--enable_diagnostics', action='store_true', help="是否啟動 Mask 儲存與困難樣本萃取")
    parser.add_argument('--need_negative', action='store_true', help="是否需要使用負樣本 (加入此參數即代表 True)")

    args = parser.parse_args()

    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = UnifiedInferencePipeline(config, device, mil_method=args.mil_method)
    pipeline.execute(args.method, args.field, file_name=args.file_name, text_prompt=args.text_prompt,
                         need_negative=args.need_negative,enable_diagnostics=args.enable_diagnostics)