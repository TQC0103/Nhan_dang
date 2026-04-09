from .anchor_free_head import AnchorFreeHead
from .anchor_head import AnchorHead
from .atss_head import ATSSHead
from .fcos_head import FCOSHead
from .gfl_head import GFLHead
from .scrfd_head import SCRFDHead

try:
    from .centripetal_head import CentripetalHead
except Exception:
    CentripetalHead = None
try:
    from .corner_head import CornerHead
except Exception:
    CornerHead = None
try:
    from .fovea_head import FoveaHead
except Exception:
    FoveaHead = None
try:
    from .free_anchor_retina_head import FreeAnchorRetinaHead
except Exception:
    FreeAnchorRetinaHead = None
try:
    from .fsaf_head import FSAFHead
except Exception:
    FSAFHead = None
try:
    from .ga_retina_head import GARetinaHead
except Exception:
    GARetinaHead = None
try:
    from .ga_rpn_head import GARPNHead
except Exception:
    GARPNHead = None
try:
    from .guided_anchor_head import FeatureAdaption, GuidedAnchorHead
except Exception:
    FeatureAdaption = None
    GuidedAnchorHead = None
try:
    from .nasfcos_head import NASFCOSHead
except Exception:
    NASFCOSHead = None
try:
    from .paa_head import PAAHead
except Exception:
    PAAHead = None
try:
    from .pisa_retinanet_head import PISARetinaHead
except Exception:
    PISARetinaHead = None
try:
    from .pisa_ssd_head import PISASSDHead
except Exception:
    PISASSDHead = None
try:
    from .reppoints_head import RepPointsHead
except Exception:
    RepPointsHead = None
try:
    from .retina_head import RetinaHead
except Exception:
    RetinaHead = None
try:
    from .retina_sepbn_head import RetinaSepBNHead
except Exception:
    RetinaSepBNHead = None
try:
    from .rpn_head import RPNHead
except Exception:
    RPNHead = None
try:
    from .sabl_retina_head import SABLRetinaHead
except Exception:
    SABLRetinaHead = None
try:
    from .ssd_head import SSDHead
except Exception:
    SSDHead = None
try:
    from .transformer_head import TransformerHead
except Exception:
    TransformerHead = None
try:
    from .vfnet_head import VFNetHead
except Exception:
    VFNetHead = None
try:
    from .yolact_head import YOLACTHead, YOLACTProtonet, YOLACTSegmHead
except Exception:
    YOLACTHead = None
    YOLACTProtonet = None
    YOLACTSegmHead = None
try:
    from .yolo_head import YOLOV3Head
except Exception:
    YOLOV3Head = None

__all__ = [
    'AnchorFreeHead', 'AnchorHead', 'ATSSHead', 'FCOSHead', 'GFLHead',
    'SCRFDHead'
]
for _name in [
        'GuidedAnchorHead', 'FeatureAdaption', 'RPNHead', 'GARPNHead',
        'RetinaHead', 'RetinaSepBNHead', 'GARetinaHead', 'SSDHead',
        'RepPointsHead', 'FoveaHead', 'FreeAnchorRetinaHead', 'FSAFHead',
        'NASFCOSHead', 'PISARetinaHead', 'PISASSDHead', 'CornerHead',
        'YOLACTHead', 'YOLACTSegmHead', 'YOLACTProtonet', 'YOLOV3Head',
        'PAAHead', 'SABLRetinaHead', 'CentripetalHead', 'VFNetHead',
        'TransformerHead']:
    if globals().get(_name) is not None:
        __all__.append(_name)
