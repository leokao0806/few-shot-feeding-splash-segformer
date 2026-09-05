import torch
from torch import nn, Tensor
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalNLLLoss(nn.Module):
    def __init__(self, ignore_label: int = 255, gamma: float = 2.0, alpha=None, weight: torch.Tensor = None,
                 aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.gamma = 2.0 if gamma is None else gamma
        self.aux_weights = aux_weights
        self.weight = weight  # optional for NLL
        self.alpha = torch.tensor(alpha) if isinstance(alpha, (list, tuple)) else alpha

    def _forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        preds = preds.clamp(min=1e-8, max=1.0)
        log_preds = torch.log(preds)
        ce_loss = F.nll_loss(
            log_preds, labels,
            weight=self.weight.to(preds.device) if self.weight is not None else None,
            ignore_index=self.ignore_label,
            reduction='none'
        )

        # --- 🛡️ 新增防爆盾牌：將 255 暫時視為 0 進行數學運算 ---
        safe_labels = labels.clone()
        safe_labels[labels == self.ignore_label] = 0

        # --- 替換：這裡使用 safe_labels 取代原本的 labels ---
        pt = safe_labels * preds[:, 1, :, :] + (1 - safe_labels) * preds[:, 0, :, :]
        focal_term = (1 - pt) ** self.gamma

        # alpha weighting
        if self.alpha is not None:
            # 替換：這裡也使用 safe_labels
            alpha_t = self.alpha * safe_labels + (1 - self.alpha) * (1 - safe_labels)
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        # 最後的守門員：255 區域的數字依然會在這裡被完全丟棄！
        valid_mask = (labels != self.ignore_label)
        return focal_loss[valid_mask].mean()

    def forward(self, preds, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(p, labels) for (p, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)

class NLLLoss(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        # self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        self.criterion = nn.NLLLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        preds = preds.clamp(min=1e-8, max=1.0)
        preds = torch.log(preds)
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class SubclassLoss(nn.Module):
    def __init__(self,
                 topk_ratio: float = 0.1,  # 正樣本中前 10% 的像素作為 Witness
                 tau: float = 0.5,  # 存在性門檻：最高信心值至少要達到 0.5
                 lambda_pos: float = 1.0,  # 正樣本權重
                 lambda_neg: float = 1.0,  # 負樣本權重
                 lambda_penalty: float = 2.0,  # 防止逃避的懲罰權重
                 ignore_label: int = 255):
        super().__init__()
        self.topk_ratio = topk_ratio
        self.tau = tau
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.lambda_penalty = lambda_penalty
        self.ignore_label = ignore_label

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        preds: [B, 2, H, W] (模型輸出的結果，包含負樣本與正樣本通道)
        labels: [B, H, W] (Ground Truth)
        """
        # 取得正樣本通道的機率值 (經過 Sigmoid 或直接從 result[1] 拿)
        # 注意：我們在 Unified 模型中輸出的 y 是 cat([neg, pos])
        pos_probs = preds[:, 1, :, :]  # [B, H, W]

        loss_total = 0.0
        batch_size = labels.size(0)

        for i in range(batch_size):
            label = labels[i]
            prob = pos_probs[i]

            # 建立 Mask
            pos_mask = (label == 1)
            neg_mask = (label == 0)
            valid_pos_count = pos_mask.sum()

            # --- 1. 正樣本損失 (Multiple-choice / Top-K Witness) ---
            if valid_pos_count > 0:
                pos_values = prob[pos_mask]
                # 計算要取多少點作為 Top-K
                k = max(1, int(valid_pos_count * self.topk_ratio))
                topk_values, _ = torch.topk(pos_values, k)

                # 物理意義：只要這 Top-K 個點中獎就好，目標趨近於 1
                # 使用 Binary Cross Entropy 邏輯
                pos_loss = -torch.log(topk_values + 1e-8).mean()

                # --- 2. 存在性懲罰 (Existential Penalty - 防止逃避) ---
                # 物理意義：如果這張圖正樣本最強的點都低於 tau，代表模型在逃避，給予重罰
                max_val = topk_values[0]  # topk 已排序，第一個最大
                if max_val < self.tau:
                    penalty = (self.tau - max_val) ** 2
                    pos_loss += self.lambda_penalty * penalty
            else:
                pos_loss = 0.0

            # --- 3. 負樣本損失 (Total Suppression) ---
            neg_values = prob[neg_mask]
            if neg_values.numel() > 0:
                # 物理意義：負樣本每一個都要趨近於 0
                neg_loss = -torch.log(1 - neg_values + 1e-8).mean()
            else:
                neg_loss = 0.0

            loss_total += (self.lambda_pos * pos_loss + self.lambda_neg * neg_loss)

        return loss_total / batch_size


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        # self.criterion = nn.NLLLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        # preds = torch.log(preds + 1e-20)
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


__all__ = ['FocalNLLLoss','NLLLoss', 'CrossEntropy', 'OhemCrossEntropy', 'Dice', 'SubclassLoss']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)