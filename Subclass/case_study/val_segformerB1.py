import torch
from tqdm import tqdm
from semseg.metrics import Metrics

@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn):
    """
    原版 SegFormer 專用評估函式
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
        # 原版 forward pass
        output = model(images)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # 計算驗證 Loss
        vloss = loss_fn(logits, labels)

        # 更新指標
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