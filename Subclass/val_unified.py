import torch
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F

# 基礎組件
from semseg.augmentations import get_val_augmentation
from semseg.metrics import Metrics

@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn):
    """
    Stage 1 專用評估函式：固定關閉 Q' (q_prime=False)，確保執行 L2 正規化。
    """
    model.eval()
    n_classes = dataloader.dataset.n_classes
    ignore_label = dataloader.dataset.ignore_label
    metrics = Metrics(n_classes, ignore_label, device)

    iter_cnt = 0
    valid_loss = 0.0
    loss_err = []

    for batch_data in tqdm(dataloader, desc="Evaluating", leave=False):
        # 取得 RGB 影像與標籤
        images, labels = batch_data[0], batch_data[1]
        iter_cnt += 1
        images = images.to(device)
        labels = labels.to(device)

        # --- 執行模型推論 ---
        # 🔶 關鍵：固定 q_prime=False。
        # 這會觸發 unified Block 內的: subclass = F.normalize(out, p=2)
        output = model(images, q_prime=False)

        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output

        # 計算驗證 Loss (對齊學長的 NLLLoss 邏輯)
        # 注意：如果 loss_fn 是 NLLLoss，它內部會處理 torch.log(logits)
        vloss = loss_fn(logits, labels)

        # 更新指標 (preds 預期為機率分佈)
        preds = logits.softmax(dim=1)
        metrics.update(preds, labels)

        valid_loss += vloss.item()
        loss_err.append(vloss.item())

    # 計算統計結果
    valid_loss = valid_loss / iter_cnt
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    return acc, macc, f1, mf1, ious, miou, valid_loss, loss_err