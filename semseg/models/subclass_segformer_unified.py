import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
# 確保引入的是我們剛才重構的 Unified 版本 Head
from semseg.models.heads import SubclassSegFormerHead_Unified


class SubclassSegFormer_Unified(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 2,
                 subclass: int = 64, total_field: int = 5, num_prompts: int = 1) -> None:
        super().__init__(backbone, num_classes)

        # 取得 Backbone 輸出通道數
        embed_dim = 256 if 'B0' in backbone or 'B1' in backbone else 768

        # 初始化整合版解碼頭
        self.decode_head = SubclassSegFormerHead_Unified(
            dims=self.backbone.channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            subclass=subclass,
            total_field=total_field,
            num_prompts=num_prompts
        )
        self.apply(self._init_weights)

    def forward(self, x: Tensor, subclass_weight=None, q_prime=False) -> Tensor:
        """
        物理路徑控制：
        1. subclass_weight -> MIL 點/袋提示 (L2 Normalization) [cite: 12]
        2. q_prime=True -> Q-prime 路徑 (Raw Logits)
        """
        y = self.backbone(x)

        y, subclass = self.decode_head(
            y,
            subclass_weight=subclass_weight,
            q_prime=q_prime
        )

        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y, subclass


if __name__ == '__main__':
    # 測試程式碼
    # 模擬 512x512 的輸入影像
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SubclassSegFormer_Unified('MiT-B0', subclass=64).to(device)
    x = torch.randn(1, 3, 512, 512).to(device)

    # 測試模式 A: 自動環境感知
    y_auto, s_auto = model(x)
    print(f"Auto Mode - Output: {y_auto.shape}, Subclass: {s_auto.shape}")

    # 測試模式 B: Q-prime 文字提示模式
    y_q, s_q = model(x, q_prime=True)
    print(f"Q-prime Mode - Output: {y_q.shape}, Subclass Logits Unique count: {len(s_q.unique())}")

    # 測試模式 C: 學長 MIL 提示模式 (假設外部給予權重)
    mock_weight = torch.randn(64).to(device)  # 簡化示意，實際應為優化後的權重
    # y_mil, s_mil = model(x, subclass_weight=mock_weight)