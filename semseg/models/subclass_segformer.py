import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SubclassSegFormerHead


class SubclassSegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 2, subclass: int = 64, kernel_size: int = 3, loop: int = 0) -> None:
        super().__init__(backbone, num_classes)
        # self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.decode_head = SubclassSegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes, subclass, kernel_size, loop)
        self.apply(self._init_weights)

    def forward(self, x: Tensor, subclass_weight=None) -> Tensor:
        y = self.backbone(x)
        if subclass_weight is None:
            y, subclass = self.decode_head(y)
        else:
            y, subclass = self.decode_head(y, subclass_weight)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y, subclass


if __name__ == '__main__':
    model = SubclassSegFormer('MiT-B0')
    # model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    y = torch.softmax(y, dim=1)
    print(y.shape, y.unique())