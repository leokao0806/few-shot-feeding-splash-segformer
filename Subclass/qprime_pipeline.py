import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from pathlib import Path
import numpy as np

from semseg.losses import get_loss
from semseg.metrics import Metrics


class QprimePipeline:
    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model
        self.device = device

        self.prompt_cfg = cfg.get('PROMPT_LEARNING', {})
        self.epochs = self.prompt_cfg.get('EPOCHS', 150)
        self.lr = self.prompt_cfg.get('MAPPER_LR', 0.005)
        self.weight_decay = self.prompt_cfg.get('WEIGHT_DECAY', 0.01)

        self.train_loss_fn = get_loss(cfg['LOSS']['NAME'])
        self.val_loss_fn = get_loss(cfg['EVAL']['LOSS_NAME'])

    def fit(self, train_loader, val_loader, field_name):
        """
        微調 Q-prime：使用傳入的 DataLoader (現在 A~E 的 RAG 樣本) 進行優化
        """
        # 1. 凍結其餘參數，僅優化 q_prime 與 gamma
        params_to_optimize = []
        for name, param in self.model.named_parameters():
            # ⬇️ 修改點 1：將 gamma 加入優化名單
            if 'q_prime' in name or 'gamma' in name:
                param.requires_grad = True
                params_to_optimize.append(param)
            else:
                param.requires_grad = False

        optimizer = AdamW(params_to_optimize, lr=self.lr, weight_decay=self.weight_decay)

        best_iou = -1.0
        best_state = None
        history = {'train_loss': [], 'val_iou': []}

        print(f"✅ [Tuning] 開始微調 Q-prime 模式")
        print(f"   - 目標場域: {field_name}")
        print(f"   - 提示來源: Graph RAG 檢索之已知場域樣本")

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                optimizer.zero_grad()
                preds, _ = self.model(imgs, q_prime=True)

                # 尺寸對齊
                if preds.shape[-2:] != lbls.shape[-2:]:
                    preds = F.interpolate(preds, size=lbls.shape[-2:], mode='bilinear', align_corners=False)

                loss = self.train_loss_fn(preds, lbls)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # 驗證 (使用 RAG 提供的 Validation 樣本)
            val_iou, _ = self._evaluate(val_loader)
            history['val_iou'].append(val_iou)

            if val_iou > best_iou:
                best_iou = val_iou
                # ⬇️ 修改點 2：同時把 gamma 的最佳狀態存下來
                best_state = {
                    'q_prime': self.model.decode_head.subclass_block.q_prime.detach().clone(),
                    'gamma': self.model.decode_head.subclass_block.gamma.detach().clone(),
                }

            if (epoch + 1) % 50 == 0 or epoch == 0:
                # 順便印出目前的 gamma 值，方便你觀察老師提議的比例變化
                current_gamma = self.model.decode_head.subclass_block.gamma.item()
                # 因為 forward 裡面有 clamp，顯示 clamp 後的實際作用值比較直觀
                clamped_gamma = max(0.0, min(1.0, current_gamma))
                print(f"   Epoch [{epoch + 1}/{self.epochs}] | Loss: {avg_train_loss:.4f} | RAG-Val IoU: {val_iou:.4f} | Gamma: {clamped_gamma:.3f}")

        print(f"✅ 微調完成。最佳 RAG-Val IoU: {best_iou:.4f}")
        self._apply_best_state(best_state)
        return best_state, history

    def _apply_best_state(self, best_state):
        if best_state:
            self.model.decode_head.subclass_block.q_prime.data = best_state['q_prime'].to(self.device)
            # ⬇️ 修改點 3：訓練結束後，把表現最好的 gamma 還原回去
            if 'gamma' in best_state:
                self.model.decode_head.subclass_block.gamma.data = best_state['gamma'].to(self.device)

    @torch.no_grad()
    def _evaluate(self, val_loader):
        self.model.eval()
        n_classes = 2
        ignore_label = getattr(val_loader.dataset, 'ignore_label', 255)
        metrics = Metrics(n_classes, ignore_label, self.device)
        total_loss = 0

        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(self.device), lbls.to(self.device)
            preds, _ = self.model(imgs, q_prime=True)

            if preds.shape[-2:] != lbls.shape[-2:]:
                preds = F.interpolate(preds, size=lbls.shape[-2:], mode='bilinear', align_corners=False)

            loss = self.val_loss_fn(preds, lbls)
            total_loss += loss.item()
            metrics.update(preds.softmax(dim=1), lbls)

        ious, _ = metrics.compute_iou()
        return ious[1], total_loss / len(val_loader)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import AdamW
# from pathlib import Path
# import numpy as np
#
# from semseg.losses import get_loss
# from semseg.metrics import Metrics
#
#
# class QprimePipeline:
#     def __init__(self, cfg, model, device):
#         self.cfg = cfg
#         self.model = model
#         self.device = device
#
#         self.prompt_cfg = cfg.get('PROMPT_LEARNING', {})
#         self.epochs = self.prompt_cfg.get('EPOCHS', 150)
#         self.lr = self.prompt_cfg.get('MAPPER_LR', 0.005)
#         self.weight_decay = self.prompt_cfg.get('WEIGHT_DECAY', 0.01)
#
#         self.train_loss_fn = get_loss(cfg['LOSS']['NAME'])
#         self.val_loss_fn = get_loss(cfg['EVAL']['LOSS_NAME'])
#
#     def fit(self, train_loader, val_loader, field_name):
#         """
#         微調 Q-prime：使用傳入的 DataLoader (現在 A~E 的 RAG 樣本) 進行優化
#         """
#         # 1. 凍結其餘參數，僅優化 q_prime
#         params_to_optimize = []
#         for name, param in self.model.named_parameters():
#             if 'q_prime' in name:
#                 param.requires_grad = True
#                 params_to_optimize.append(param)
#             else:
#                 param.requires_grad = False
#
#         optimizer = AdamW(params_to_optimize, lr=self.lr, weight_decay=self.weight_decay)
#
#         best_iou = -1.0
#         best_state = None
#         history = {'train_loss': [], 'val_iou': []}
#
#         print(f"✅ [Tuning] 開始微調 Q-prime 模式")
#         print(f"   - 目標場域: {field_name}")
#         print(f"   - 提示來源: Graph RAG 檢索之已知場域樣本")
#
#         for epoch in range(self.epochs):
#             self.model.train()
#             total_train_loss = 0
#
#             for imgs, lbls in train_loader:
#                 imgs, lbls = imgs.to(self.device), lbls.to(self.device)
#                 optimizer.zero_grad()
#                 preds, _ = self.model(imgs, q_prime=True)
#
#                 # 尺寸對齊
#                 if preds.shape[-2:] != lbls.shape[-2:]:
#                     preds = F.interpolate(preds, size=lbls.shape[-2:], mode='bilinear', align_corners=False)
#
#                 loss = self.train_loss_fn(preds, lbls)
#                 loss.backward()
#                 optimizer.step()
#                 total_train_loss += loss.item()
#
#             avg_train_loss = total_train_loss / len(train_loader)
#             history['train_loss'].append(avg_train_loss)
#
#             # 驗證 (使用 RAG 提供的 Validation 樣本)
#             val_iou, _ = self._evaluate(val_loader)
#             history['val_iou'].append(val_iou)
#
#             if val_iou > best_iou:
#                 best_iou = val_iou
#                 best_state = {
#                     'q_prime': self.model.decode_head.subclass_block.q_prime.detach().clone(),
#                 }
#
#             if (epoch + 1) % 50 == 0 or epoch == 0:
#                 print(f"   Epoch [{epoch + 1}/{self.epochs}] | Loss: {avg_train_loss:.4f} | RAG-Val IoU: {val_iou:.4f}")
#
#         print(f"✅ 微調完成。最佳 RAG-Val IoU: {best_iou:.4f}")
#         self._apply_best_state(best_state)
#         return best_state, history
#
#     def _apply_best_state(self, best_state):
#         if best_state:
#             self.model.decode_head.subclass_block.q_prime.data = best_state['q_prime'].to(self.device)
#
#     @torch.no_grad()
#     def _evaluate(self, val_loader):
#         self.model.eval()
#         n_classes = 2
#         ignore_label = getattr(val_loader.dataset, 'ignore_label', 255)
#         metrics = Metrics(n_classes, ignore_label, self.device)
#         total_loss = 0
#
#         for imgs, lbls in val_loader:
#             imgs, lbls = imgs.to(self.device), lbls.to(self.device)
#             preds, _ = self.model(imgs, q_prime=True)
#
#             if preds.shape[-2:] != lbls.shape[-2:]:
#                 preds = F.interpolate(preds, size=lbls.shape[-2:], mode='bilinear', align_corners=False)
#
#             loss = self.val_loss_fn(preds, lbls)
#             total_loss += loss.item()
#             metrics.update(preds.softmax(dim=1), lbls)
#
#         ious, _ = metrics.compute_iou()
#         return ious[1], total_loss / len(val_loader)