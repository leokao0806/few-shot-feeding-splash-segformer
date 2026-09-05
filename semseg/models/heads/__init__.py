from .upernet import UPerHead
from .segformer import SegFormerHead
from .sfnet import SFHead
from .fpn import FPNHead
from .fapn import FaPNHead
from .fcn import FCNHead
from .condnet import CondHead
from .lawin import LawinHead
from .subclass import SubclassSegFormerHead
from .subclass import SubclassSegFormerHead_visual_prompt
from .subclass import SubclassSegFormerHead_Unified

__all__ = ['UPerHead', 'SegFormerHead', 'SFHead', 'FPNHead', 'FaPNHead', 'FCNHead', 'CondHead', 'LawinHead',
           'SubclassSegFormerHead','SubclassSegFormerHead_visual_prompt', 'SubclassSegFormerHead_Unified']