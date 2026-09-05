import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SubclassSegFormerHead_visual_prompt


class SubclassSegFormer_visual_prompt(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 2, subclass: int = 64, num_prompts: int = 1) -> None:
        super().__init__(backbone, num_classes)
        # 確保 Head 使用新定義的版本 (內含自由變數 q_prime)
        self.decode_head = SubclassSegFormerHead_visual_prompt(
            self.backbone.channels,
            256 if 'B0' in backbone or 'B1' in backbone else 768,
            num_classes, subclass,
            num_prompts=num_prompts
        )
        self.apply(self._init_weights)

    # 【關鍵修改 1】：接收 prompt_enabled 參數 (預設 False)
    def forward(self, x: Tensor, prompt_enabled=False) -> Tensor:
        y = self.backbone(x)  # y 是一個 Tuple (4 層特徵)

        # 【關鍵修改 2】：將開關往下傳遞給 decode_head
        y, subclass = self.decode_head(y, prompt_enabled=prompt_enabled)

        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y, subclass


if __name__ == '__main__':
    model = SubclassSegFormer_visual_prompt('MiT-B0')
    # model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    y = torch.softmax(y, dim=1)
    print(y.shape, y.unique())